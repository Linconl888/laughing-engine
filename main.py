import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from tqdm import tqdm
from model import Transformer
from config import get_config
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_data
from transformers import logging, AutoTokenizer, AutoModel

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PyTorch sees devices:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert_chinese':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            base_model = AutoModel.from_pretrained('bert-base-chinese')
        else:
            raise ValueError('unknown model')
        self.model = Transformer(base_model, args.num_classes, args.method)
        self.model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        self.model.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            #n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
            predicts = torch.argmax(outputs['predicts'], -1)
            n_correct += (predicts == targets).sum().item()
            n_train += targets.size(0)
        return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        wrong_texts = []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                #n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
                predicts = torch.argmax(outputs['predicts'], -1)
                n_correct += (predicts == targets).sum().item()
                n_test += targets.size(0)
                
                wrong_mask = predicts != targets
                if wrong_mask.any():
                    for i in wrong_mask.nonzero(as_tuple=True)[0]:
                        ids = inputs['input_ids'][i].cpu().tolist()
                        text = self.tokenizer.decode(ids, skip_special_tokens=True)
                        wrong_texts.append(
                            f'GroudTrue:{targets[i].item()} Predicts:{predicts[i].item()} | {text[:80]}...'
                        )
    
        if wrong_texts:
            print(f'\n>>> WrongExample(All number: {len(wrong_texts)}):')
            for wt in wrong_texts[:10]:
                print(f'  {wt}')
            
        return test_loss / n_test, n_correct / n_test

    def run(self):
        train_dataloader, test_dataloader, label_word_lengths = load_data(dataset=self.args.dataset,
                                                                          data_dir=self.args.data_dir,
                                                                          tokenizer=self.tokenizer,
                                                                          train_batch_size=self.args.train_batch_size,
                                                                          test_batch_size=self.args.test_batch_size,
                                                                          model_name=self.args.model_name,
                                                                          method=self.args.method,
                                                                          workers=0,
                                                                          seed=self.args.seed)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
            self.model.set_label_word_lengths(label_word_lengths)
        else:
            raise ValueError('unknown method')
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)
        best_loss, best_acc = float('inf'), 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc = self._test(test_dataloader, criterion)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
            self.logger.info('{}/{} - {:.2f}%'.format(epoch+1, self.args.num_epoch, 100*(epoch+1)/self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc*100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc*100))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc*100))
        self.logger.info('log saved: {}'.format(self.args.log_name))


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()
