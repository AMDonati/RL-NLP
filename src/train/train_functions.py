import time
import torch
import torch.nn.functional as F

def assert_correctness_batch(inputs, targets):
    assert torch.all(torch.eq(inputs[:,1:], targets[:,:-1])) == 1, "error in inputs/targets"


def train_one_epoch(model, train_generator, optimizer, criterion, device, grad_clip=None, print_interval=10):
    model.train()  # Turns on train mode which enables dropout.
    total_loss = 0.
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(train_generator):
        inputs = inputs.to(device)
        targets = targets.view(targets.size(1) * targets.size(0)).to(device)  # targets (S*B)
        model.zero_grad()
        output, hidden = model(inputs)  # output (S * B, V), hidden (num_layers,B,1)
        loss = criterion(output, targets)
        loss.backward()
        # clip grad norm:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
        # print loss every number of batches
        if (batch + 1) % print_interval == 0:
            print('loss for batch {}: {:5.3f}'.format(batch + 1, total_loss / (batch + 1)))
            print('time for {} training steps: {:5.2f}'.format(print_interval, time.time() - start_time))

    curr_loss = total_loss / (batch + 1)
    elapsed = time.time() - start_time

    return curr_loss, elapsed


def train_one_epoch_vqa(model, train_generator, optimizer, criterion, device, grad_clip=None, print_interval=10):
    model.train()  # Turns on train mode which enables dropout.
    total_loss = 0.
    start_time = time.time()
    for batch, ((inputs, targets), _, _) in enumerate(train_generator):
        inputs = inputs.to(device)
        targets = targets.view(targets.size(1) * targets.size(0)).to(device)  # targets (S*B)
        model.zero_grad()
        output, hidden = model(inputs)  # output (S * B, V), hidden (num_layers,B,1)
        loss = criterion(output, targets)
        loss.backward()
        # clip grad norm:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
        # print loss every number of batches
        if (batch + 1) % print_interval == 0:
            print('loss for batch {}: {:5.3f}'.format(batch + 1, total_loss / (batch + 1)))
            print('time for {} training steps: {:5.2f}'.format(print_interval, time.time() - start_time))

    curr_loss = total_loss / (batch + 1)
    elapsed = time.time() - start_time

    return curr_loss, elapsed


def train_one_epoch_policy(model, train_generator, optimizer, criterion, device, grad_clip, print_interval=10):
    model.train()  # Turns on train mode which enables dropout.
    total_loss = 0.
    start_time = time.time()
    for batch, ((inputs, targets), answers, img) in enumerate(train_generator):
        if isinstance(img, list):
            feats = img[0]
        else:
            feats = img
        inputs, feats, answers = inputs.to(device), feats.to(device), answers.to(device)
        targets = targets.view(targets.size(1) * targets.size(0)).to(device)  # targets (S*B)
        model.zero_grad()
        logits, _ = model(inputs, feats, answers)  # output (S * B, V)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = criterion(log_probs, targets)
        loss.backward()
        # clip grad norm:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
        # print loss every number of batches
        if (batch + 1) % print_interval == 0:
            print('loss for batch {}: {:5.3f}'.format(batch + 1, total_loss / (batch + 1)))
            print('time for {} training steps: {:5.2f}'.format(print_interval, time.time() - start_time))

    curr_loss = total_loss / (batch + 1)
    elapsed = time.time() - start_time

    return curr_loss, elapsed


def evaluate(model, val_generator, criterion, device):
    model.eval()  # turn on evaluation mode which disables dropout.
    total_loss = 0.
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(val_generator):
            inputs = inputs.to(device)
            targets = targets.view(targets.size(1) * targets.size(0)).to(device)
            output, hidden = model(inputs)
            total_loss += criterion(output, targets).item()

    return total_loss / (batch + 1)


def evaluate_vqa(model, val_generator, criterion, device):
    model.eval()  # turn on evaluation mode which disables dropout.
    total_loss = 0.
    with torch.no_grad():
        for batch, ((inputs, targets), _, _) in enumerate(val_generator):
            inputs = inputs.to(device)
            targets = targets.view(targets.size(1) * targets.size(0)).to(device)
            output, hidden = model(inputs)
            total_loss += criterion(output, targets).item()

    return total_loss / (batch + 1)


def evaluate_policy(model, val_generator, criterion, device):
    model.eval()  # turn on evaluation mode which disables dropout.
    total_loss = 0.
    with torch.no_grad():
        for batch, ((inputs, targets), answers, img) in enumerate(val_generator):
            if isinstance(img, list):
                feats = img[0]
            else:
                feats = img
            inputs, feats, answers = inputs.to(device), feats.to(device), answers.to(device)
            targets = targets.view(targets.size(1) * targets.size(0)).to(device)
            logits, _ = model(inputs, feats, answers)
            log_probs = F.log_softmax(logits, dim=-1)
            total_loss += criterion(log_probs, targets).item()

    return total_loss / (batch + 1)
