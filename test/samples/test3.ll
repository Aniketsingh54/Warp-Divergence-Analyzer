; ModuleID = 'test3.ll'
source_filename = "test3.cu"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define void @kernel3(i32* %a) !dbg !10 {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !12
  %cmp1 = icmp slt i32 %tid, 128, !dbg !13
  br i1 %cmp1, label %then1, label %else1, !dbg !14

then1:
  store i32 5, i32* %a
  br label %mid

else1:
  store i32 10, i32* %a
  br label %mid

mid:
  %cmp2 = icmp sgt i32 %tid, 256, !dbg !15
  br i1 %cmp2, label %then2, label %else2, !dbg !16

then2:
  store i32 15, i32* %a
  br label %end

else2:
  store i32 20, i32* %a
  br label %end

end:
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!7}
!llvm.ident = !{!8}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !2, producer: "warp-div-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "test3.cu", directory: "/")
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"warp-div"}
!10 = distinct !DISubprogram(name: "kernel3", scope: !2, file: !2, line: 1, type: !11, unit: !1)
!11 = !DISubroutineType(types: !{})
!12 = !DILocation(line: 3, column: 15, scope: !10)
!13 = !DILocation(line: 4, column: 10, scope: !10)
!14 = !DILocation(line: 5, column: 3, scope: !10)
!15 = !DILocation(line: 9, column: 10, scope: !10)
!16 = !DILocation(line: 10, column: 3, scope: !10)
