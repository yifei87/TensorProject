# -*- coding:utf-8 -*-
from tensorflow.python.tools.freeze_graph import freeze_graph

# input_graph='./model/trained.pb' # 这里的pb文件是用tf.train.write_graph方法保存的
input_graph=None # 模型结构

input_meta_graph='./model/trained.ckpt.meta' #模型结构

input_checkpoint='./model/trained.ckpt' # 这里若是r12以上的版本，只需给.data-00000....前面的文件名，
                                # 如：model.ckpt.1001.data-00000-of-00001，只需写model.ckpt.1001


output_graph='./model/frozen_graph5.pb'
output_node_names='hypothesis' # 这里的输出节点名为 hypothesis 而不是 softmax（由程序而定）

''' ---上面参数随程序修改，以下参数不用修改---------------------'''

input_saver=""
input_binary=True #False # 如果 False时，出现
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 47: invalid start byte
# input_binary 改成True

restore_op_name="save/restore_all"
filename_tensor_name="save/Const:0"
clear_devices=True
initializer_nodes=''


freeze_graph(
    input_graph=input_graph,
    input_meta_graph=input_meta_graph,
    input_saver=input_saver,
    input_binary=input_binary,
    input_checkpoint=input_checkpoint,
    output_node_names=output_node_names,
    restore_op_name=restore_op_name,
    filename_tensor_name=filename_tensor_name,
    output_graph=output_graph,
    clear_devices=clear_devices,
    initializer_nodes=initializer_nodes,
)

