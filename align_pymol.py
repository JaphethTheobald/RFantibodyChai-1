python
import pymol
objs = cmd.get_object_list()
ref = objs[0]
for obj in objs[1:]:
    cmd.align(obj, ref)
python end