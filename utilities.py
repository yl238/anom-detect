import time
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def trainVAE(train_loader, model, criterion, optimizer, epoch, args):
    """
    Iterate through the train data and perform optimization
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_avg = AverageMeter()
    kl_avg = AverageMeter()
    reconst_logp_avg = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda()

        recon_batch, mu, logvar = model(input)
        loss, loss_details = criterion(recon_batch, input, mu, logvar)

        # record loss
        loss_avg.update(loss.item(), input.size(0))
        kl_avg.update(loss_details['KL'].item(), input.size(0))
        reconst_logp_avg.update(loss_details['reconst_logp'].item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'reconst_logp {reconst_logp_avg.val:.4f} ({reconst_logp_avg.avg:.4f})\t'
                  'kl {kl_avg.val:.4f} ({kl_avg.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, reconst_logp_avg=reconst_logp_avg, kl_avg=kl_avg,
                   loss=loss_avg))

    return loss_avg.avg, kl_avg.avg, reconst_logp_avg.avg


def validateVAE(val_loader, model, criterion, args):
    """
    iterate through the validate set and output the accuracy
    """
    batch_time = AverageMeter()
    loss_avg = AverageMeter()
    kl_avg = AverageMeter()
    reconst_logp_avg = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, _) in enumerate(val_loader):
        if args.cuda:
            input = input.cuda()

        # compute output
        recon_batch, mu, logvar = model(input)
        loss, loss_details = criterion(recon_batch, input, mu, logvar)

        # measure accuracy and record loss
        loss_avg.update(loss.item(), input.size(0))
        kl_avg.update(loss_details['KL'].item(), input.size(0))
        reconst_logp_avg.update(loss_details['reconst_logp'].item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'reconst_logp {reconst_logp_avg.val:.4f} ({reconst_logp_avg.avg:.4f})\t'
                  'kl {kl_avg.val:.4f} ({kl_avg.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, reconst_logp_avg=reconst_logp_avg,
                   kl_avg=kl_avg, loss=loss_avg))
    return loss_avg.avg, kl_avg.avg, reconst_logp_avg.avg

def evaluateVAE(test_loader, model, criterion, args):
    """
    iterate through test loader and find out average loss of normal and
    abnormal
    """
    avg_abnormal_loss = AverageMeter()
    avg_normal_loss = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in tqdm(enumerate(test_loader)):
       if args.cuda:
           input = input.cuda()

       # compute output
       recon_batch, mu, logvar = model(input)
       loss, loss_details = criterion(recon_batch, input, mu, logvar)

       # if normal
       if target.item() == 1:
           avg_normal_loss.update(loss.item(), input.size(0))
       else:
           avg_abnormal_loss.update(loss.item(), input.size(0))

    return avg_normal_loss.avg, avg_abnormal_loss.avg


