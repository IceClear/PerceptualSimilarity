import imageio
import subprocess
import glob
import os

method_name = ['ori', 'li', 'cvpr', 'dcad', 'dncnn', 'dscnn', 'arcnn', 'MFQE2.0']
# method_name = ['dscnn']
QP='32'
gpu_id='0,1'
nof = 0

for method in method_name:
    p = subprocess.Popen(['python','compute_dists_dirs.py','-d0','/media/iceclear/IceKing2/compare_results/QP_32_test/gt','-d1','/media/iceclear/IceKing2/compare_results/QP_32_test/'+method])
    p.communicate()
    print('>>>>>>>>>>>>>>'+method+'finish')
