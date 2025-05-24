; ModuleID = 'test1.ll'
source_filename = "test1.cu"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define void @kernel1(i32* %a) !dbg !10 {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !12
  %mod = srem i32 %tid, 2, !dbg !13
  %cmp = icmp eq i32 %mod, 0, !dbg !14
  br i1 %cmp, label %then, label %else, !dbg !15

then:
  %idx1 = getelementptr inbounds i32, i32* %a, i32 %tid
  store i32 1, i32* %idx1
  br label %end

else:
  %idx2 = getelementptr inbounds i32, i32* %a, i32 %tid
  store i32 2, i32* %idx2
  br label %end

end:
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!7}
!llvm.ident = !{!8}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !2, producer: "warp-div-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "test1.cu", directory: "/")
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"warp-div"}
!10 = distinct !DISubprogram(name: "kernel1", scope: !2, file: !2, line: 1, type: !11, unit: !1)
!11 = !DISubroutineType(types: !{})
!12 = !DILocation(line: 3, column: 15, scope: !10)
!13 = !DILocation(line: 4, column: 13, scope: !10)
!14 = !DILocation(line: 5, column: 10, scope: !10)
!15 = !DILocation(line: 6, column: 3, scope: !10)
