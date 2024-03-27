objects = allocation_protocol_generated.h \
		  cudevice_attributes_generated.h \
		  trace_execution_protocol_generated.h \
		  cuda_runtime_calls_generated.h \
		  cublas_calls_generated.h \
		  cudnn_calls_generated.h \

all: $(objects)

%_generated.h: %.fbs
	./flatc-2 --cpp $<
	sed -i 's/names\[122\]/names\[128\]/' cudevice_attributes_generated.h # temporary flatbuffer bugfix

clean:
	rm $(objects)
