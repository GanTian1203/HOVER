"""Optional CPU integration checks for the pinned external source trees."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hover.data import Windows
from hover.io import load, save, sha256
from hover.method import HOVER
from hover.training import fit

from test_science import REPORT
import test_science


class SourceRequirements:
    def setUp(self):
        root = Path(os.environ.get("HOVER_EXTERNAL_ROOT", "external_models"))
        if not (root / self.source).is_dir():
            self.skipTest(
                f"Missing pinned {self.source} source under {root}. "
                "See docs/EXTERNAL_MODEL_GUIDE.md and set HOVER_EXTERNAL_ROOT."
            )


class LinearIntegration(SourceRequirements, unittest.TestCase):
    source = "LTSF"
    fixture = test_science.Science.fixture

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

class FITSIntegration(SourceRequirements, unittest.TestCase):
    source = "FITS"

    def test_fits_source_identity(self):
        from hover.backbones import external_source
        external_source('FITS',None)
        from hover.io import config as source_config
        bad_source=dict(source_config('sources')['FITS']);bad_source['files']={'models/FITS.py':'0'*64}
        with patch('hover.backbones.config',return_value={'FITS':bad_source}):
            with self.assertRaises(RuntimeError):external_source('FITS',None)
