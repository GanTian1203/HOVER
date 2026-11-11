"""Acquire immutable official source files after exact content verification."""
import io
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile
import hashlib

from .io import Blocked, config, save, sha256


def verify(name, destination):
    from .backbones import external_source, source_key
    key=source_key(name);root=external_source(name,destination);source=config("sources")[key]
    checked=sum(p.endswith('.py') or p in source['license_files'] for p in source['files'])
    return dict(status="PINNED_SOURCE_FILES_PASS",name=name,source_key=key,commit=source["commit"],files_verified=checked,license=source["license"],source_directory=str(root))


def register(name, local_path, destination, commit, permission_basis=None):
    from .backbones import source_key
    key=source_key(name);source=config("sources")[key];root=Path(local_path).resolve()
    if commit != source["commit"]:raise Blocked("Local source requires the exact pinned --commit")
    if key=="OLINEAR" and not permission_basis:raise Blocked("Document the actual basis for your own use;this is not a redistribution grant")
    for path,expected in source["files"].items():
        p=root/path
        if path.endswith(".py") or path in source["license_files"]:
            if not p.is_file() or sha256(p)!=expected:raise Blocked("Local source identity mismatch: "+path)
    import subprocess
    if (root/".git").exists():
        observed=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
        if observed!=commit:raise Blocked("Local Git HEAD differs from pinned commit")
    wrapper=Path(destination)/key;wrapper.mkdir(parents=True,exist_ok=True)
    path=wrapper/("USER_SUPPLIED_SOURCE.json" if key=="OLINEAR" else "ACQUISITION.json")
    value=dict(commit=commit,source_directory=str(root),permission_basis=permission_basis or source["license"],identity_basis="exact_pinned_files_and_declared_commit;Git_HEAD_checked_if_present")
    if path.exists():
        from .io import load
        if load(path)!=value:raise Blocked("Existing registration differs;use a separate wrapper directory")
    else:save(path,value)
    return verify(name,destination)


def acquire(name, destination):
    from .backbones import source_key
    name=source_key(name)
    source = config("sources")[name]
    if source["redistribution"] == "EXTERNAL_USER_SUPPLIED_ONLY":
        raise Blocked("OLinear source permission unresolved. This tool does not download or redistribute it. Supply a separately obtained pinned source and document your permission basis.")
    destination = Path(destination) / name
    if destination.exists():
        return verify(name,destination.parent)
    with urllib.request.urlopen(source["archive_url"], timeout=60) as response:
        content = response.read(64 * 1024 * 1024 + 1)
    if len(content) > 64 * 1024 * 1024:
        raise ValueError("Source archive size limit exceeded")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if archive.comment.decode("ascii") != source["commit"]:
            raise ValueError("Archive commit identity mismatch")
        members = {}
        for item in archive.infolist():
            p = PurePosixPath(item.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in item.filename:
                raise ValueError("Unsafe archive path")
            if len(p.parts) > 1 and not item.is_dir():
                members[PurePosixPath(*p.parts[1:]).as_posix()] = item
        selected = {}
        for path, expected in source["files"].items():
            if path not in members:
                raise ValueError("Frozen source file absent: " + path)
            blob = archive.read(members[path])
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError("Source file differs from experiment: " + path)
            selected[path] = blob
        destination.mkdir(parents=True, exist_ok=False)
        for path, blob in selected.items():
            p = destination / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
    result = dict(status="PINNED_ARCHIVE_AND_FILES_PASS",name=name, commit=source["commit"], files_verified=len(selected), license=source["license"])
    save(destination / "ACQUISITION.json", result)
    return result
