第 1 步：保留 GSM8K Dataset，确认 128 样本 SFT 能稳定跑完

第 2 步：把 max_examples 改成参数，跑 128/256/512/1024/full

第 3 步：写 eval 脚本，对每个 checkpoint 算 validation accuracy

第 4 步：画 validation accuracy curve

第 5 步：最后再加 wandb

第 6 步：如果必须严格贴合作业要求，再把 eval 从普通 generate 换成 vLLM


  当前状态干净，main 已经跟踪 origin/main。__pycache__/ 和 .pyc 不会被 Git 管理。

  以后你本地改完代码，只需要：

  cd "/Users/fengjubo/Desktop/stanford cs336/assignment5-alignment-main/homework"
  git add .
  git commit -m "你的更新说明"
  git push

  服务器上第一次用：

  cd 你的assignment5父目录
  mv homework homework.bak
  git clone https://github.com/fengjubo/cs336_assignment5.git homework

  之后服务器每次跑之前更新：

  cd 你的assignment5父目录/homework
  git pull --ff-only

  或者在父目录下：

  git -C homework pull --ff-only

  • 可以，服务器上如果也改了 homework/，流程就是：服务器 commit + push，本地 pull。

  服务器上：

  cd 你的assignment5父目录/homework

  git status
  git add .
  git commit -m "update from autodl"
  git push

  然后本地同步：

  cd "/Users/fengjubo/Desktop/stanford cs336/assignment5-alignment-main/homework"

  git pull --ff-only

  如果你本地也改了，还没提交，先在本地提交再 pull：

  git add .
  git commit -m "local update"
  git pull --rebase
  git push

  推荐你养成一个习惯：

  - 去服务器跑之前：本地 git push
  - 服务器开始跑之前：服务器 git pull --ff-only
  - 服务器上改了代码：服务器 git add . && git commit -m "..." && git push
  - 回本地继续改之前：本地 git pull --ff-only

  如果只是服务器产生了日志、checkpoint、wandb/ 这些运行结果，不要提交。我们已经在 .gitignore 里忽略了一部分训练输出目录。