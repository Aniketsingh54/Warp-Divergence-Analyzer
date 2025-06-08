; ModuleID = 'kernels.bc'
source_filename = "kernels.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@flag = dso_local local_unnamed_addr addrspace(3) global i32 undef, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define dso_local void @kernel_if_else(ptr nocapture noundef writeonly %0) local_unnamed_addr #0 {
  %2 = tail call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %3 = and i32 %2, 1
  %4 = icmp eq i32 %3, 0
  %5 = zext nneg i32 %2 to i64
  %6 = getelementptr inbounds i32, ptr %0, i64 %5
  %7 = mul nuw nsw i32 %2, 3
  %8 = shl nuw nsw i32 %2, 1
  %9 = select i1 %4, i32 %8, i32 %7
  store i32 %9, ptr %6, align 4, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define dso_local void @kernel_nested_branch(ptr nocapture noundef writeonly %0) local_unnamed_addr #0 {
  %2 = tail call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %3 = icmp ult i32 %2, 16
  br i1 %3, label %4, label %11

4:                                                ; preds = %1
  %5 = and i32 %2, 3
  %6 = icmp eq i32 %5, 0
  %7 = zext nneg i32 %2 to i64
  %8 = getelementptr inbounds i32, ptr %0, i64 %7
  br i1 %6, label %9, label %10

9:                                                ; preds = %4
  store i32 10, ptr %8, align 4, !tbaa !10
  br label %14

10:                                               ; preds = %4
  store i32 20, ptr %8, align 4, !tbaa !10
  br label %14

11:                                               ; preds = %1
  %12 = zext nneg i32 %2 to i64
  %13 = getelementptr inbounds i32, ptr %0, i64 %12
  store i32 30, ptr %13, align 4, !tbaa !10
  br label %14

14:                                               ; preds = %9, %10, %11
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define dso_local void @kernel_shared_flag(ptr nocapture noundef writeonly %0) local_unnamed_addr #1 {
  %2 = tail call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  store i32 1, ptr addrspacecast (ptr addrspace(3) @flag to ptr), align 4, !tbaa !10
  br label %5

5:                                                ; preds = %4, %1
  tail call void @llvm.nvvm.barrier0()
  %6 = load i32, ptr addrspacecast (ptr addrspace(3) @flag to ptr), align 4, !tbaa !10
  %7 = icmp eq i32 %6, 1
  %8 = zext nneg i32 %2 to i64
  %9 = getelementptr inbounds i32, ptr %0, i64 %8
  %10 = select i1 %7, i32 100, i32 200
  %11 = add nuw nsw i32 %2, %10
  store i32 %11, ptr %9, align 4, !tbaa !10
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_75" "target-features"="+ptx83,+sm_75" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_75" "target-features"="+ptx83,+sm_75" "uniform-work-group-size"="true" }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4, !5, !6}
!llvm.ident = !{!7, !8}
!nvvmir.version = !{!9}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 3]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{ptr @kernel_if_else, !"kernel", i32 1}
!5 = !{ptr @kernel_nested_branch, !"kernel", i32 1}
!6 = !{ptr @kernel_shared_flag, !"kernel", i32 1}
!7 = !{!"Ubuntu clang version 19.1.7 (++20250114103320+cd708029e0b2-1~exp1~20250114103432.75)"}
!8 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!9 = !{i32 2, i32 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C++ TBAA"}
