import unittest
import sys
import os.path as path
import os
from Stoner.compat import listdir_recursive
from importlib import import_module
import matplotlib.pyplot as plt
from traceback import format_exc

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)

class DocSamples_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"doc","samples")

    def setUp(self):
        self.scripts=[path.relpath(x,self.datadir).replace(path.sep,".") for x in listdir_recursive(self.datadir,"*.py") if not x.endswith("__init__.py")]
        sys.path.insert(0,self.datadir)
        
    def test_scripts(self):
        """Import each of the sample scripts in turn and see if they ran without error"""
        failures=[]
        for ix,filename in enumerate(self.scripts):
            os.chdir(self.datadir)
            script=filename[:-3]
            print("Trying script {}: {}".format(ix,filename))
            try:
                os.chdir(self.datadir)
                code=import_module(script)
                fignum=len(plt.get_fignums())
                self.assertGreaterEqual(fignum,1,"{} Did not produce any figures !".format(script))
                print("Done")
                fig=plt.gcf()
                plt.close(fig)
                plt.close("all")
            except Exception:
                v=format_exc()
                print("Failed with\n{}".format(v))
                failures.append("Script file {} failed with {}".format(filename,v))
        self.assertTrue(len(failures)==0,"\n".join(failures))
                
if __name__=="__main__": # Run some tests manually to allow debugging
    test=DocSamples_test("test_scripts")
    test.setUp()
    test.test_scripts()
                
            
