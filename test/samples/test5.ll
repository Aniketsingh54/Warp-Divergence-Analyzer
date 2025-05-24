; Attach debug info to instructions
define void @kernel5(i32* %a) !dbg !6 {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !10
  %cond = icmp eq i32 %tid, 7, !dbg !11
  br i1 %cond, label %then, label %else, !dbg !12

then:                                             ; preds = %entry
  store i32 111, i32* %a, align 4, !dbg !13
  br label %exit, !dbg !14

else:                                             ; preds = %entry
  store i32 222, i32* %a, align 4, !dbg !15
  br label %exit, !dbg !16

exit:                                             ; preds = %then, %else
  ret void, !dbg !17
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !5, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !{}, retainedTypes: !{}, globals: !{}, imports: !{})
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{!"clang"}
!5 = !DIFile(filename: "test5.cu", directory: "/tmp")
!6 = distinct !DISubprogram(name: "kernel5", scope: !5, file: !5, line: 1, type: null, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1)

!10 = !DILocation(line: 3, column: 5, scope: !6)
!11 = !DILocation(line: 4, column: 10, scope: !6)
!12 = !DILocation(line: 5, column: 5, scope: !6)
!13 = !DILocation(line: 7, column: 7, scope: !6)
!14 = !DILocation(line: 8, column: 5, scope: !6)
!15 = !DILocation(line: 10, column: 7, scope: !6)
!16 = !DILocation(line: 11, column: 5, scope: !6)
!17 = !DILocation(line: 13, column: 5, scope: !6)
