import time
import torch

def repackage_hidden(h):
  """Wraps hidden states in new Tensors, to detach them from their history.
  so that each hidden_state h_t is detached from the backprop graph once used. """
  if isinstance(h, torch.Tensor):
    return h.detach()
  else:
    return tuple(repackage_hidden(v) for v in h)


def train_one_epoch(model, train_generator, optimizer, criterion, device, BATCH_SIZE, args, print_interval=10):
  model.train()  # Turns on train mode which enables dropout.
  hidden = model.init_hidden(BATCH_SIZE)
  total_loss = 0.
  start_time = time.time()

  # loop over batches
  for batch, (inputs, targets) in enumerate(train_generator):
    inputs, targets = inputs.to(device), targets.to(device)
    inputs, targets = inputs.long().t(), targets.view(targets.size(1) * targets.size(0)).long()  # inputs: (S,B) # targets: (S*B)
    optimizer.zero_grad()
    hidden = repackage_hidden(hidden)
    output, hidden = model(inputs, hidden)  # output (S * B, V), hidden (S,B,1)
    loss = criterion(output, targets)
    loss.backward()

    # clip grad norm:
    if args.grad_clip is not None:
      torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.grad_clip)
    optimizer.step()
    total_loss += loss.item()

    # print loss every number of batches
    if (batch + 1) % print_interval == 0:
      print('loss for batch {}: {:5.3f}'.format(batch + 1, total_loss / (batch + 1)))
      print('time for {} training steps: {:5.2f}'.format(print_interval, time.time() - start_time))

  curr_loss = total_loss / (batch + 1)
  elapsed = time.time() - start_time

  return curr_loss, elapsed


def evaluate(model, val_generator, criterion, device, BATCH_SIZE):
  model.eval()  # turn on evaluation mode which disables dropout.
  total_loss = 0.
  hidden = model.init_hidden(BATCH_SIZE)
  with torch.no_grad():
    for batch, (inputs, targets) in enumerate(val_generator):
      inputs, targets = inputs.to(device), targets.to(device)
      inputs, targets = inputs.long().t(), targets.view(targets.size(1) * targets.size(0)).long()
      output, hidden = model(inputs, hidden)
      hidden = repackage_hidden(hidden)
      total_loss += criterion(output, targets).item()

  return total_loss / (batch + 1)