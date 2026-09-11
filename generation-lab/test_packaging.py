import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"web/backend"))
from pack_asset import pack_asset, write_arrays


class PackagingTests(unittest.TestCase):
    def fixture(self, folder):
        directions=np.zeros((2,3,3),dtype="<f4");directions[:,1,:]=[[.1,0,.2],[0,-.3,0]]
        arrays={"expressionDirections":directions,"skinBones":np.array([0,1,1],dtype="<u2"),"skinWeights":np.array([1,.3,.7],dtype="<f4")}
        poses=[]
        for i in range(7):
            poses.append({"name":str(i)})
            arrays[f"reference_{i}_positions"]=np.zeros((3,3),dtype="<f4")
            arrays[f"reference_{i}_rotations"]=np.zeros((3,4),dtype="<f4")
        meta=dict(version=1,deformation="lhm-linear-blend-v1",nExpressions=2,nGaussians=3,referencePoses=poses)
        meta.update(write_arrays(folder,"avatar.bin",arrays));(folder/"avatar.json").write_text(json.dumps(meta))
        return meta,arrays

    def test_numeric_values_and_skinning_preserved_references_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source";source.mkdir();meta,arrays=self.fixture(source)
            result=pack_asset(source,root/"packed")
            self.assertEqual(result["version"],2);self.assertEqual(result["nExpressionVertices"],1)
            self.assertFalse(any(k.startswith("reference_") for k in result["arrays"]))
            self.assertEqual(len(result["validation"]["arrays"]),14)
            payload=(root/"packed/avatar.bin").read_bytes();self.assertEqual(hashlib.sha256(payload).hexdigest(),result["sha256"])
            for name in ("skinBones","skinWeights"):
                spec=result["arrays"][name];self.assertEqual(payload[spec["offset"]:spec["offset"]+spec["bytes"]],arrays[name].tobytes())
            self.assertEqual(hashlib.sha256((source/"avatar.bin").read_bytes()).hexdigest(),meta["sha256"])
            with self.assertRaises(ValueError):pack_asset(source,root/"packed")
            with self.assertRaises(ValueError):pack_asset(source,source)

    def test_corrupt_source_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source";source.mkdir();meta,arrays=self.fixture(source)
            meta["sha256"]="0"*64;(source/"avatar.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):pack_asset(source,root/"packed")
            self.assertFalse((root/"packed").exists())

    def test_pending_manifest_stays_hidden_until_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source";source.mkdir();self.fixture(source)
            pack_asset(source,root/"pending",manifest_name="avatar.pending.json")
            self.assertFalse((root/"pending/avatar.json").exists())
            self.assertTrue((root/"pending/avatar.pending.json").is_file())
            with self.assertRaises(ValueError):pack_asset(source,root/"pending",manifest_name="avatar.pending.json")


if __name__=="__main__":unittest.main()
