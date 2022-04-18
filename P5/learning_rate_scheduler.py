def halving_scheduler_4(epoch, lr):
  if epoch % 4 == 0 and epoch != 0:
    print(f"We've halved the lr to be {0.5 * lr}")
    return 0.5 * lr
  else:
    return lr

def halving_scheduler_10(epoch, lr):
  if epoch % 4 == 0 and epoch != 0:
    print(f"We've halved the lr to be {0.5 * lr}")
    return 0.5 * lr
  else:
    return lr
