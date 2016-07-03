nameI=flowc
nameC=flowcmodule
pyv=2.7
swig -python $nameI.i
gcc -fpic -c $nameC.c ${nameI}_wrap.c -I/usr/include/python$pyv/
gcc -shared $nameC.o ${nameI}_wrap.o -o _$nameI.so
