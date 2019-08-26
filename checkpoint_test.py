
from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file('./weights/flappy_bird/flappy_bird.ckpt', tensor_name='', all_tensors=True)

