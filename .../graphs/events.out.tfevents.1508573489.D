       �K"	  @L�z�Abrain.Event:2��!      ��M�	�i[L�z�A"�
G
ConstConst*
value	B :*
dtype0*
_output_shapes
: 
l
summand1
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
summand1/AssignAssignsummand1Const*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@summand1
a
summand1/readIdentitysummand1*
_output_shapes
: *
T0*
_class
loc:@summand1
I
Const_1Const*
dtype0*
_output_shapes
: *
value	B :
l
summand2
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
summand2/AssignAssignsummand2Const_1*
T0*
_class
loc:@summand2*
validate_shape(*
_output_shapes
: *
use_locking(
a
summand2/readIdentitysummand2*
T0*
_class
loc:@summand2*
_output_shapes
: 
I
sumAddsummand1/readsummand2/read*
T0*
_output_shapes
: 
0
initNoOp^summand1/Assign^summand2/Assign
2
init_1NoOp^summand1/Assign^summand2/Assign"\�I��      ���	�[L�z�AJ�
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

NoOp
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.3.02v1.3.0-rc2-20-g0787eee�
G
ConstConst*
value	B :*
dtype0*
_output_shapes
: 
l
summand1
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
�
summand1/AssignAssignsummand1Const*
T0*
_class
loc:@summand1*
validate_shape(*
_output_shapes
: *
use_locking(
a
summand1/readIdentitysummand1*
_output_shapes
: *
T0*
_class
loc:@summand1
I
Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
l
summand2
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
summand2/AssignAssignsummand2Const_1*
T0*
_class
loc:@summand2*
validate_shape(*
_output_shapes
: *
use_locking(
a
summand2/readIdentitysummand2*
T0*
_class
loc:@summand2*
_output_shapes
: 
I
sumAddsummand1/readsummand2/read*
T0*
_output_shapes
: 
0
initNoOp^summand1/Assign^summand2/Assign
2
init_1NoOp^summand1/Assign^summand2/Assign""o
	variablesb`
.

summand1:0summand1/Assignsummand1/read:0
.

summand2:0summand2/Assignsummand2/read:0"y
trainable_variablesb`
.

summand1:0summand1/Assignsummand1/read:0
.

summand2:0summand2/Assignsummand2/read:0�K*�