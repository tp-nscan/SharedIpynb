       �K"	  ��&��Abrain.Event:2ݝU�o      -��R	%T��&��A"�
Q
x/initial_valueConst*
dtype0*
_output_shapes
: *
value	B :
e
x
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
x/AssignAssignxx/initial_value*
use_locking(*
T0*
_class

loc:@x*
validate_shape(*
_output_shapes
: 
L
x/readIdentityx*
T0*
_class

loc:@x*
_output_shapes
: 
Q
y/initial_valueConst*
value	B :*
dtype0*
_output_shapes
: 
e
y
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
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
;
mulMulx/readx/read*
T0*
_output_shapes
: 
:
mul_1Mulmuly/read*
T0*
_output_shapes
: 
:
addAddmul_1y/read*
T0*
_output_shapes
: 
I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
;
add_1Addaddadd_1/y*
_output_shapes
: *
T0"PB�      ��̠	�_��&��AJ�
��
9
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.3.02v1.3.0-rc2-20-g0787eee�
Q
x/initial_valueConst*
dtype0*
_output_shapes
: *
value	B :
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
x/AssignAssignxx/initial_value*
T0*
_class

loc:@x*
validate_shape(*
_output_shapes
: *
use_locking(
L
x/readIdentityx*
T0*
_class

loc:@x*
_output_shapes
: 
Q
y/initial_valueConst*
value	B :*
dtype0*
_output_shapes
: 
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
y/AssignAssignyy/initial_value*
T0*
_class

loc:@y*
validate_shape(*
_output_shapes
: *
use_locking(
L
y/readIdentityy*
T0*
_class

loc:@y*
_output_shapes
: 
;
mulMulx/readx/read*
T0*
_output_shapes
: 
:
mul_1Mulmuly/read*
T0*
_output_shapes
: 
:
addAddmul_1y/read*
T0*
_output_shapes
: 
I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
;
add_1Addaddadd_1/y*
T0*
_output_shapes
: ""E
	variables86

x:0x/Assignx/read:0

y:0y/Assigny/read:0"O
trainable_variables86

x:0x/Assignx/read:0

y:0y/Assigny/read:0|�5�