; ModuleID = 'test2.ll'
source_filename = "test2.cu"

define void @kernel2(i32* %a, i32 %flag) !dbg !10 {
entry:
  %cmp = icmp eq i32 %flag, 0, !dbg !12
  br i1 %cmp, label %then, label %else, !dbg !13

then:
  store i32 1, i32* %a
  br label %end

else:
  store i32 2, i32* %a
  br label %end

end:
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!7}
!llvm.ident = !{!8}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !2, producer: "warp-div-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "test2.cu", directory: "/")
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"warp-div"}
!10 = distinct !DISubprogram(name: "kernel2", scope: !2, file: !2, line: 1, type: !11, unit: !1)
!11 = !DISubroutineType(types: !{})
!12 = !DILocation(line: 3, column: 10, scope: !10)
!13 = !DILocation(line: 4, column: 3, scope: !10)
