def halving_scheduler(epoch, lr):
  if epoch % 4 == 0 and epoch != 0:
    print(f"We've halved the lr to be {0.5 * lr}")
    return 0.5 * lr
  else:
    return lr
