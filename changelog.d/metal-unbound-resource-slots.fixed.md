Metal output gives each buffer, texture and sampler without a binding the
lowest free slot of its binding space. It used to start them one past the
highest explicit binding, so a kernel whose constants block is bound at buffer 3
took its unbound storage buffers at buffers 4 and 5 instead of 0 and 1 (#3010).
