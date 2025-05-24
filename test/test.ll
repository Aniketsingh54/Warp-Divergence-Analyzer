define void @kernel(i32* %a) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %rem = srem i32 %tid, 2
  %cond = icmp eq i32 %rem, 0
  br i1 %cond, label %then, label %else

then:
  ret void

else:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
