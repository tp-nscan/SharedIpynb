       �K"	  @�'��Abrain.Event:2�7��Y      Kd�:	�!b�'��A"�
Q
x/initial_valueConst*
value	B :*
dtype0*
_output_shapes
: 
e
x
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
x/AssignAssignxx/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class

loc:@x
L
x/readIdentityx*
T0*
_class

loc:@x*
_output_shapes
: 
Q
y/initial_valueConst*
dtype0*
_output_shapes
: *
value	B :
e
y
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
y/AssignAssignyy/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class

loc:@y
L
y/readIdentityy*
T0*
_class

loc:@y*
_output_shapes
: 
:
sqMulx/readx/read*
T0*
_output_shapes
: 
C
cConst*
value	B :*
dtype0*
_output_shapes
: 
7
mulMulsqy/read*
T0*
_output_shapes
: 
8
addAddmuly/read*
T0*
_output_shapes
: 
1
fAddaddc*
T0*
_output_shapes
: "��