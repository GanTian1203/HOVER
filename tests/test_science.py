import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

import reference_graph
from hover.analysis import pair, placements, records_from_csv, summarize
from hover.data import Windows, calculate_q, time_features, validate_q
from hover.experiments import task_matrix
from hover.io import identity, load, save, sha256
from hover.method import HOVER, Simple, VARIANTS, adapter_loss
from hover.training import choose, evaluate, fit, state_digest

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
REPORT = []
ARRAYS = {}


class Base(nn.Module):
    def __init__(self, horizon=96):
        super().__init__()
        self.layer = nn.Linear(336, horizon)

    def forward(self, x, xmark=None, future_mark=None):
        return self.layer(x.transpose(1, 2)).transpose(1, 2)


class Science(unittest.TestCase):
    def test_original_graph_initialization_forward_gradient_adam(self):
        generator = torch.Generator().manual_seed(20261107)
        x, y = torch.randn(2, 336, 3, generator=generator), torch.randn(2, 96, 3, generator=generator)
        for seed in range(951, 956):
            for variant in (*VARIANTS, "simple"):
                torch.manual_seed(17)
                base = Base()
                original = reference_graph.SIMPLE(base, seed) if variant == "simple" else reference_graph.HOVER(base, seed, variant)
                public = Simple(base, seed) if variant == "simple" else HOVER(base, seed, variant)
                self.assertEqual(set(original.state_dict()), set(public.state_dict()))
                self.assertTrue(all(torch.equal(original.state_dict()[k], public.state_dict()[k]) for k in original.state_dict()))
                parameters = sum(p.numel() for p in public.parameters() if p.requires_grad)
                if variant == "full": self.assertEqual(parameters, 42464)
                if variant == "simple": self.assertEqual(parameters, 42096)
                before = state_digest(public.base)
                opt_a = torch.optim.Adam([p for p in original.parameters() if p.requires_grad], lr=.001)
                opt_b = torch.optim.Adam([p for p in public.parameters() if p.requires_grad], lr=.001)
                maxima = []
                for update in range(3):
                    original.train(); public.train()
                    pa, ba, da = original.components(x)
                    pb, bb, db = public.components(x)
                    self.assertTrue(torch.equal(pa, pb))
                    self.assertTrue(torch.equal(ba, bb))
                    self.assertTrue(all(torch.equal(da[k], db[k]) for k in da))
                    if update == 0: self.assertTrue(torch.equal(pb, bb))
                    la = (pa-y).square().mean() + 3. * (pa-ba).square().mean()
                    lb, _, _ = adapter_loss(pb, y, bb, 3.)
                    self.assertTrue(torch.equal(la, lb))
                    opt_a.zero_grad(); opt_b.zero_grad()
                    la.backward(); lb.backward()
                    grads_a = dict(original.named_parameters()); grads_b = dict(public.named_parameters())
                    for key, parameter in grads_a.items():
                        if parameter.grad is None:
                            self.assertIsNone(grads_b[key].grad)
                        else:
                            self.assertTrue(torch.equal(parameter.grad, grads_b[key].grad))
                    torch.nn.utils.clip_grad_norm_([p for p in original.parameters() if p.requires_grad], 1.)
                    torch.nn.utils.clip_grad_norm_([p for p in public.parameters() if p.requires_grad], 1.)
                    opt_a.step(); opt_b.step()
                    self.assertTrue(all(torch.equal(original.state_dict()[k], public.state_dict()[k]) for k in original.state_dict()))
                    maxima.append(float((pa-pb).abs().max()))
                self.assertEqual(before, state_digest(public.base))
                self.assertFalse(public.base.training)
                self.assertTrue(all(p.grad is None for p in public.base.parameters()))
                ARRAYS[f"{seed}_{variant}_original"] = pa.detach().numpy()
                ARRAYS[f"{seed}_{variant}_public"] = pb.detach().numpy()
                REPORT.append(dict(case="frozen_graph", seed=seed, variant=variant, parameters=parameters,
                                   forward_max_abs=max(maxima), gradient_max_abs=0., Adam_state_max_abs=0., updates=3))
        REPORT.append(dict(case="input", generator_seed=20261107, x_shape=list(x.shape), y_shape=list(y.shape), dtype="float32"))
        ARRAYS['core_input_x'] = x.numpy()
        ARRAYS['core_target_y'] = y.numpy()

    def test_horizon_specific_heads(self):
        import ast
        import inspect
        for h in (12, 24, 48, 192, 336, 720):
            tree = ast.parse(inspect.getsource(reference_graph.Model))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 96:
                    node.value = h
            env = dict(vars(reference_graph))
            exec(compile(tree, "frozen_horizon_oracle", "exec"), env)
            env["Original"] = env["Model"]
            exec(compile(inspect.getsource(reference_graph.HOVER), "frozen_wrapper_oracle", "exec"), env)
            src = inspect.getsource(reference_graph.SIMPLE).replace("nn.Linear(432,70)", "nn.Linear(%d,70)" % (336+h)).replace("nn.Linear(70,96)", "nn.Linear(70,%d)" % h)
            exec(compile(src, "frozen_simple_horizon_oracle", "exec"), env)
            torch.manual_seed(3); base = Base(h)
            for arm, original, public in (("HOVER", env["HOVER"](base, 731), HOVER(base, 731, horizon=h)),
                                           ("SIMPLE", env["SIMPLE"](base, 731), Simple(base, 731, horizon=h))):
                self.assertTrue(all(torch.equal(original.state_dict()[k], public.state_dict()[k]) for k in original.state_dict()))
                x = torch.randn(2, 336, 2)
                # Exercise the residual heads beyond their zero initialization.
                with torch.no_grad():
                    for key, tensor in original.state_dict().items():
                        if ("head" in key) and tensor.is_floating_point():
                            tensor.add_(.01)
                    public.load_state_dict(original.state_dict())
                self.assertTrue(torch.equal(original(x), public(x)))
                self.assertEqual(public(x).shape, (2,h,2))
                ARRAYS[f"H{h}_{arm}_x"] = x.numpy()
                ARRAYS[f"H{h}_{arm}_original"] = original(x).detach().numpy()
                ARRAYS[f"H{h}_{arm}_public"] = public(x).detach().numpy()
                REPORT.append(dict(case="horizon", horizon=h, arm=arm, forward_max_abs=0., parameters=sum(p.numel() for p in public.parameters() if p.requires_grad)))
        with self.assertRaises(ValueError): HOVER(Base(24), 731)(torch.randn(2,336,2))

    def fixture(self, root, raw_member=False):
        generator = np.random.default_rng(733)
        raw = generator.normal(size=(900,3)).astype(np.float64)
        mu, sd = raw[:650].mean(0), raw[:650].std(0)
        arrays = dict(z=((raw-mu)/sd).astype(np.float32), mu=mu, sd=sd,
                      dates=np.datetime64("2016-07-01T00")+np.arange(900)*np.timedelta64(1,"h"))
        if raw_member: arrays["raw"] = raw
        p = root/"ETTh1.npz"; np.savez_compressed(p, **arrays)
        definition = {"ETTh1":dict(channels=3,TRAIN_rows=[0,650],DEV_rows=[650,900])}
        save(p.with_suffix(".json"),dict(dataset="ETTh1",TRAIN_rows=[0,650],DEV_rows=[650,900],cache_SHA256=sha256(p),TEST_access=False))
        return p, raw, arrays, definition

    def test_data_scaler_roles_calendar_and_q_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, raw, arrays, definition = self.fixture(root)
            formal = Windows("ETTh1",p,24,"formal",definitions=definition)
            self.assertTrue(np.array_equal(formal.z.numpy(),arrays["z"]))
            self.assertTrue(np.allclose(formal.inverse(formal.z.numpy()),raw,rtol=0,atol=2e-7))
            for key, data in (("formal",formal),("inner",Windows("ETTh1",p,24,"inner",definitions=definition))):
                for role in ("TRAIN","DEV"):
                    x,y,xm,ym=data.batch(role,[0,len(data.origins[role])-1])
                    for j,index in enumerate([0,-1]):
                        origin=data.origins[role][index]
                        self.assertTrue(torch.equal(x[j],data.z[origin-336:origin]))
                        self.assertTrue(torch.equal(y[j],data.z[origin:origin+24]))
                with self.assertRaises(PermissionError): data.batch("TEST",[0])
            inner=Windows("ETTh1",p,24,"inner",definitions=definition)
            expected=arrays["z"][:650].astype(np.float64)
            expected=((expected-expected[:520].mean(0))/expected[:520].std(0)).astype(np.float32)
            self.assertTrue(np.array_equal(expected,inner.z.numpy()))
            self.assertEqual(time_features(arrays["dates"])[0].tolist(),[-.5,.1666666716337204,-.5,-.001369863050058484])
            paths={};hashes={}
            for key,n in (("history",336),("future",24)):
                path=root/(key+".npy");np.save(path,np.eye(n));paths[key]=str(path);hashes[key]=sha256(path)
            q=dict(inner.manifest(),paths=paths,SHA256=hashes)
            validate_q(q,inner)
            for field,value in (("dataset","ECL"),("role","formal"),("horizon",12),("history",96),("cache_SHA256","bad"),("statistics_rows",[0,650])):
                with self.assertRaises(ValueError): validate_q(dict(q,**{field:value}),inner)
            bad=copy.deepcopy(q);bad["SHA256"]["future"]="0"*64
            with self.assertRaises(ValueError): validate_q(bad,inner)
            with self.assertRaises(ValueError): Windows("ETTh1",p,24,"TEST",definitions=definition)
            with self.assertRaises(ValueError): calculate_q(np.zeros((450,2)),12)
            raw[:,1]=0
            actual,skip=calculate_q(raw,12)
            covariance=[]
            for j in (0,2):
                matrix=np.cov(np.lib.stride_tricks.sliding_window_view(raw[:,j],12).T)
                covariance.append((matrix/np.diag(matrix)).astype(np.float32))
            _,vectors=np.linalg.eigh(np.mean(covariance,axis=0))
            self.assertTrue(np.array_equal(actual,np.flip(vectors.T,axis=0)))
            self.assertEqual(skip,[1])
            REPORT.append(dict(case="data_roles_calendar_Q",inner_fit_end=520,formal_train_end=650,all_negative_role_checks="PASS",Q_max_abs=0.))

    def test_fit_recovery_and_wrong_identity(self):
        from unittest.mock import patch
        import hover.training as training
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,_,_,defs=self.fixture(root)
            data=Windows("ETTh1",p,24,"inner",definitions=defs)
            torch.manual_seed(10);base=Base(24)
            model_a=HOVER(base,731,horizon=24);model_b=HOVER(base,731,horizon=24)
            protocol=dict(lr=.001,retention=3.,batch=64,epochs=2,minimum_epochs=2,patience=3,clip=1.,lradj="constant",objective="adapter",selection="best_validation")
            binding=dict(dataset="SYNTHETIC",kind="CPU_CONFORMANCE_ONLY")
            a=fit(model_a,data,protocol,731,root/"whole",binding,"NLINEAR_PAIRED",base_identity={"SHA256":"synthetic"})
            original=training.atomic_torch
            def fail_after_boundary(path,value):
                original(path,value)
                if Path(path).name=="resume.pt" and value["epoch"]==1: raise OSError("injected CPU test interruption")
            with patch.object(training,"atomic_torch",fail_after_boundary):
                with self.assertRaises(OSError): fit(model_b,data,protocol,731,root/"resume",binding,"NLINEAR_PAIRED",base_identity={"SHA256":"synthetic"})
            b=fit(model_b,data,protocol,731,root/"resume",binding,"NLINEAR_PAIRED",base_identity={"SHA256":"synthetic"})
            self.assertTrue(all(torch.equal(model_a.state_dict()[k],model_b.state_dict()[k]) for k in model_a.state_dict()))
            self.assertEqual(a["metrics"],b["metrics"])
            with self.assertRaises(ValueError): fit(model_b,data,protocol,732,root/"resume",binding,"NLINEAR_PAIRED",base_identity={"SHA256":"synthetic"})
            with self.assertRaises(ValueError): fit(model_b,data,protocol,731,root/"resume",dict(binding,horizon=12),"NLINEAR_PAIRED",base_identity={"SHA256":"synthetic"})
            REPORT.append(dict(case="epoch_resume",synthetic_updates=a["updates"],final_state_max_abs=0.,metrics_difference=0.,wrong_identity="REJECTED"))

    def test_prefix_preparation_and_schema_rejection(self):
        import gzip
        from unittest.mock import patch
        from hover.prepare import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            definition = dict(channels=3, TRAIN_rows=[0,650], DEV_rows=[650,900],
                keep_raw=False, format='gzip_csv', source_full_SHA256=None, column_order='all',
                accepted_cache_SHA256='synthetic', nominal_minutes=60)
            source = root/'source.gz'
            with gzip.open(source,'wb') as stream:
                for i in range(900):
                    stream.write(('%s,%s,%s\n' % (i+1,i%7,i*.1)).encode())
                stream.write(b'SEALED_SUFFIX_MUST_NOT_BE_NUMERICALLY_PARSED\n')
            with patch('hover.prepare.config',return_value={'ETTh1':definition}):
                result = prepare('ETTh1',source,root/'prepared.npz')
            self.assertEqual(result['rows_decoded'],900)
            self.assertFalse(result['TEST_access'])
            with np.load(root/'prepared.npz') as archive:
                self.assertEqual(archive['z'].shape,(900,3))
                self.assertTrue(np.allclose(archive['z'][:650].mean(0),0,atol=1e-6))
            # All admitted readers stop before an intentionally nonfinite suffix.
            pems = np.random.default_rng(10).normal(size=(1000,3,1)).astype('float32')
            pems[900:] = np.nan
            np.savez_compressed(root/'traffic.npz',data=pems)
            pems_definition = dict(definition,format='pems_npz',keep_raw=True)
            with patch('hover.prepare.config',return_value={'PEMS03':pems_definition}):
                result = prepare('PEMS03',root/'traffic.npz',root/'pems_prepared.npz')
            self.assertEqual(result['rows_decoded'],900)
            # Use a synthetic ETT file to check calendar features and headers.
            import datetime
            header = ['date', 'a', 'b', 'c']
            ett = root/'ett.csv'
            with ett.open('w', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(header)
                for i in range(900):
                    stamp = datetime.datetime(2000,1,1)+datetime.timedelta(hours=i)
                    writer.writerow([stamp.isoformat(),i,i%7,i*.1])
                writer.writerow(['SEALED_SUFFIX', 'not', 'numeric', 'values'])
            ett_definition = dict(definition,format='ett_csv',header=header)
            with patch('hover.prepare.config',return_value={'ETTh1':ett_definition}):
                prepared = prepare('ETTh1',ett,root/'ett_prepared.npz')
                self.assertEqual(prepared['rows_decoded'],900)
                with self.assertRaises(FileExistsError):prepare('ETTh1',ett,root/'ett_prepared.npz')
            with patch('hover.prepare.config',return_value={'ETTh1':dict(ett_definition,header=['wrong'])}):
                with self.assertRaises(ValueError):prepare('ETTh1',ett,root/'bad_header.npz')
            with patch('hover.prepare.config',return_value={'ETTh1':dict(ett_definition,nominal_minutes=15)}):
                with self.assertRaises(ValueError):prepare('ETTh1',ett,root/'bad_calendar.npz')
            REPORT.append(dict(case='prefix_readers',gzip_sealed_suffix='NOT_PARSED',PEMS_nonfinite_suffix='NOT_PARSED',real_source_values_used=0))

    def test_fit_lock_and_source_transport(self):
        import io
        from unittest.mock import patch
        import hover.training as training
        from hover.prepare import download
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            # Reentrant acquisition must fail before any fit can begin.
            def attempt(*args, **kwargs):
                with self.assertRaises(RuntimeError):
                    fit(None,None,None,731,root/'locked',{},'NLINEAR_PAIRED')
                return {'status':'LOCK_TEST_ONLY'}
            with patch.object(training,'_fit',attempt):
                self.assertEqual(fit(None,None,None,731,root/'locked',{},'NLINEAR_PAIRED')['status'],'LOCK_TEST_ONLY')
            with patch.object(training,'_fit',return_value={'status':'RELEASED'}):
                self.assertEqual(fit(None,None,None,731,root/'locked',{},'NLINEAR_PAIRED')['status'],'RELEASED')
            class Response(io.BytesIO):
                status=206
                headers={'Content-Range':'bytes 0-23/1000'}
            definition={'ETTh1':dict(source_url='https://example.invalid/synthetic',format='ett_csv',source_prefix_bytes=24,source_prefix_SHA256='0'*64)}
            with patch('hover.prepare.config',return_value=definition),patch('urllib.request.urlopen',return_value=Response(b'synthetic transport only')):
                with self.assertRaises(ValueError):download('ETTh1',root/'mismatch.csv')
            self.assertTrue((root/'mismatch.csv').exists())
            rejected=Response(b'SEALED_BODY_MUST_NOT_BE_READ');rejected.status=200
            with patch('hover.prepare.config',return_value=definition),patch('urllib.request.urlopen',return_value=rejected),patch.object(rejected,'read',side_effect=AssertionError('body read')):
                with self.assertRaises(RuntimeError):download('ETTh1',root/'range_ignored.csv')
            self.assertFalse((root/'range_ignored.csv').exists())
            REPORT.append(dict(case='fit_lock_and_transport',duplicate_fit='REJECTED',lock_release='PASS',bad_source_hash='REJECTED',network_calls=0))

    def test_user_supplied_source_identity(self):
        from unittest.mock import patch
        from hover.backbones import external_source
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wrapper=root/'OLINEAR';wrapper.mkdir()
            implementation=root/'provided';implementation.mkdir()
            code=implementation/'mock.py';code.write_text('# synthetic source gate fixture only\n')
            source={'commit':'pinned', 'files':{'mock.py':sha256(code)},'license_files':[]}
            with patch('hover.backbones.config',return_value={'OLINEAR':source}):
                with self.assertRaises(RuntimeError):external_source('OLINEAR',root)
                save(wrapper/'USER_SUPPLIED_SOURCE.json',dict(commit='wrong',permission_basis='synthetic test',source_directory='../provided'))
                with self.assertRaises(RuntimeError):external_source('OLINEAR',root)
                save(wrapper/'USER_SUPPLIED_SOURCE.json',dict(commit='pinned',permission_basis='synthetic test',source_directory='../provided'))
                self.assertEqual(external_source('OLINEAR',root),implementation.resolve())
                code.write_text('# changed source\n')
                with self.assertRaises(RuntimeError):external_source('OLINEAR',root)
            REPORT.append(dict(case='external_source_gate',wrong_commit='REJECTED',wrong_SHA='REJECTED',read_only_source_location='PASS',real_third_party_code_used=False))
        external_source('FITS',None)
        from hover.io import config as source_config
        bad_source=dict(source_config('sources')['FITS']);bad_source['files']={'models/FITS.py':'0'*64}
        with patch('hover.backbones.config',return_value={'FITS':bad_source}):
            with self.assertRaises(RuntimeError):external_source('FITS',None)

    def test_technical_smoke_contract(self):
        from unittest.mock import patch
        from hover.smoke import smoke
        from hover.io import config as real_config
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,_,_,defs=self.fixture(root)
            (root/'ETTh1.npz').write_bytes(p.read_bytes())
            save(root/'ETTh1.json',dict(dataset='ETTh1',TRAIN_rows=[0,650],DEV_rows=[650,900],cache_SHA256=sha256(root/'ETTh1.npz'),TEST_access=False))
            with patch('hover.data.config',return_value=defs):
                result=smoke('ETTh1',root,root/'technical')
                with self.assertRaises(FileExistsError):smoke('ETTh1',root,root/'technical')
            self.assertEqual(result['kind'],'TECHNICAL_SMOKE_NOT_FORMAL_FIT')
            self.assertTrue(all(r['batch']==64 and r['optimizer_updates']==1 and r['frozen_unchanged'] for r in result['rows']))
            REPORT.append(dict(case='technical_smoke',synthetic_only=True,arms=2,batch=64,checkpoint_recovery=True,formal_fits=0))

    def test_independent_checkpoint_aggregate_audit(self):
        from hover.audit import audit_fit
        from hover.backbones import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p,_,_,defs = self.fixture(root)
            data = Windows('ETTh1',p,24,'inner',definitions=defs)
            native = build('NLINEAR_PAIRED','ETTh1',731,24,data=data)
            pbase = dict(lr=.001,batch=64,epochs=1,minimum_epochs=1,patience=None,lradj='constant',objective='MSE',selection='fixed_last')
            b = fit(native,data,pbase,731,root/'native',dict(kind='SYNTHETIC'),'NLINEAR_PAIRED')
            self.assertEqual(audit_fit(root/'native')['status'],'PASS')
            adapter = HOVER(native,731,horizon=24)
            pa = dict(lr=.001,retention=1.,batch=64,epochs=1,minimum_epochs=1,patience=3,clip=1.,lradj='constant',objective='adapter',selection='best_validation')
            fit(adapter,data,pa,731,root/'adapter',dict(kind='SYNTHETIC',arm='HOVER'),'NLINEAR_PAIRED',base_identity=dict(checkpoint_SHA256=b['checkpoint_SHA256']))
            self.assertEqual(audit_fit(root/'adapter',base_directory=root/'native')['status'],'PASS')
            receipt = load(root/'adapter/receipt.json')
            receipt['metrics']['mse'] += .001
            save(root/'adapter/receipt.json',receipt)
            with self.assertRaises(ValueError): audit_fit(root/'adapter',base_directory=root/'native')
            REPORT.append(dict(case='independent_aggregate_and_frozen_checkpoint',status='PASS',tampered_metrics='REJECTED',real_fits=0))

    def test_matrices_reference_aggregation_and_pairing(self):
        self.assertEqual(len(task_matrix("A")),600)
        self.assertEqual(len(task_matrix("B")),240)
        self.assertEqual(len(task_matrix("C")),240)
        self.assertEqual(len(task_matrix("Table2")),225)
        for stage,d,h in (("C","Exchange",96),("B","ETTh1",192),("A","ETTh1",12),("Table2","Exchange",96)):
            with self.assertRaises(ValueError): task_matrix(stage,d,horizon=h)
        rows=records_from_csv(ROOT/"results/reference/main_DEV_seeds.csv")
        table=summarize(rows);awards,counts=placements(table)
        self.assertEqual(len(table),120)
        wins={r["system"]:r for r in counts}
        self.assertEqual(wins["OLINEAR_HOVER"]["combined_best"],15)
        self.assertEqual(wins["NLINEAR_HOVER"]["combined_best"],3)
        ab=records_from_csv(ROOT/"results/reference/ablation_DEV_seeds.csv","ablation")
        for control in ("history_only","no_history_direct","concatenation","history_condition_replacement"):
            paired,summary=pair(ab,"HOVER",control)
            self.assertEqual(len(paired),60);self.assertEqual(len(summary),12)
        bad=copy.deepcopy(ab);bad[0]["base_checkpoint_SHA256"]="wrong"
        with self.assertRaises(ValueError):pair(bad,"HOVER",bad[0]["system"])
        with self.assertRaises(ValueError):summarize(rows+rows[:1])
        with self.assertRaises(ValueError):summarize(rows[1:])
        # A sub-threshold gain keeps the earlier candidate.
        candidates=[dict(metrics=dict(mse=1.)),dict(metrics=dict(mse=1.-1e-9)),dict(metrics=dict(mse=.9))]
        self.assertIs(choose(candidates[:2]),candidates[0]);self.assertIs(choose(candidates),candidates[2])
        REPORT.append(dict(case="reference_aggregation",main_seed_rows=600,main_groups=120,OLINEAR_HOVER_best=15,NLINEAR_HOVER_best=3,ablation_groups=48))

    def test_synthetic_single_dataset_A_B_execution(self):
        from unittest.mock import patch
        import hover.experiments as experiments
        from hover.io import config as real_config
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p,_,_,defs=self.fixture(root)
            synthetic=root/'SYNTHETIC_CPU.npz'
            synthetic.write_bytes(p.read_bytes())
            save(synthetic.with_suffix('.json'),dict(dataset='SYNTHETIC_CPU',TRAIN_rows=[0,650],DEV_rows=[650,900],cache_SHA256=sha256(synthetic),TEST_access=False))
            definitions={'SYNTHETIC_CPU':defs['ETTh1']}
            paired={'SYNTHETIC_CPU':dict(base=dict(lr=.001,batch=32,epochs=1,minimum_epochs=1,patience=None,lradj='constant',objective='MSE',selection='fixed_last'),
                adapter=dict(lr=.001,retention=1.,batch=64,epochs=1,minimum_epochs=1,patience=3,clip=1.,lradj='constant',objective='adapter',selection='best_validation'))}
            def configs(name):
                if name=='datasets':return definitions
                if name=='paired_nlinear':return paired
                return real_config(name)
            with patch('hover.experiments.config',configs),patch('hover.data.config',configs):
                engine=experiments.Experiment(root,root/'outputs')
                a=engine.run(experiments.task_matrix('A','SYNTHETIC_CPU','NLINEAR_HOVER',951)[0])
                b=engine.run(experiments.task_matrix('B','SYNTHETIC_CPU','concatenation',951)[0])
            self.assertEqual(a['status'],'DONE');self.assertEqual(b['status'],'DONE')
            self.assertEqual(a['base_checkpoint_SHA256'],b['base_checkpoint_SHA256'])
            self.assertEqual(a['independent_audit'],'NUMERIC_AGGREGATE_AND_CHECKPOINT_PASS')
            REPORT.append(dict(case='single_dataset_pipeline',kind='SYNTHETIC_CPU_PROTOCOL_INTEGRATION',dataset='SYNTHETIC_CPU',stages=['A','B'],native_checkpoint_shared=True,real_dataset_fits=0))


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Science)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    path=Path(__import__("os").environ.get("HOVER_TEST_REPORT","CPU_conformance_report.json"))
    save(path,dict(status="PASS" if result.wasSuccessful() else "FAIL",tests_run=result.testsRun,
                   tolerance=dict(core="bitwise equality on inspected CPU/PyTorch",inverse_atol=2e-7,Q="bitwise same CPU eigensolver"),
                   cases=REPORT,CUDA_initialized=torch.cuda.is_initialized(),real_dataset_fits=0,TEST_access=False))
    arrays_path = __import__('os').environ.get('HOVER_TEST_ARRAYS')
    if arrays_path:
        np.savez_compressed(arrays_path,**ARRAYS)
    raise SystemExit(0 if result.wasSuccessful() else 1)
