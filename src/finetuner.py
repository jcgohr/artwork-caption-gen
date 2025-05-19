import src.parsers.LongCLIP_FinetuneParser as FinetuneParser
import submodules.Long_CLIP.model as longclip
import src.utils.mutate as mutate

import os
import json
import torch
# import warnings
# warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from colorama import Fore, Style
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from adabelief_pytorch import AdaBelief

"""
Long-CLIP finetuner
"""

class ArtCaptionDataset(Dataset):
    def __init__(self, metadata_or_path:str, caption_field_name:str, transform=None):
        self.transform = transform
        self.caption_field_name = caption_field_name 
        if not isinstance(metadata_or_path, dict):
            with open(metadata_or_path,'r',encoding='utf-8') as file:
                self.metadata=list(json.loads(file.read()).values())
        else:
            self.metadata = list(metadata_or_path.values())
        self.image_paths = []
        for sample in self.metadata:
            self.image_paths.append(sample["file_path"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        caption = self.metadata[idx][self.caption_field_name]
        if(isinstance(caption, list)): # If caption is list of sentences (in case of artpedia), make into single caption
            caption = " ".join(caption)
        text = longclip.tokenize(caption, truncate=True) # Tokenize the caption

        return image, text.squeeze(0) # Remove the extra dimension

class finetune:
    """
    Finetune longclip checkpoints
    """
    def __init__(self, val_or_path:str, train_or_path:str, caption_field_name:str, checkpoint_output_path:str, epochs:int=6, batch_size:int=40, save_min_loss:bool=False, early_stop:bool=False, checkpoint_input_path:str='submodules/longclip/checkpoints/longclip-L.pt', **kwargs):
        """
        Args:
            val_path: Path to validation set, or already loaded set
            train_path: Path to train set, or already loaded set
            caption_field_name: Name of caption field within metadata file
            checkpoint_output_path: Desired path to output finetuned checkpoint files
            epochs: How many epochs to finetune for
            save_min_loss: Only save checkpoints when the validation loss is lower than all previous checkpoints
            checkpoint_input_path: Path of checkpoints to be finetuned
        """
        self.caption_field_name = caption_field_name
        self.checkpoint_input_path = checkpoint_input_path
        self.checkpoint_output_path = checkpoint_output_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_min_loss = save_min_loss
        self.early_stop = early_stop

        # Save training plots with matplotlib to:
        self.plots_folder = kwargs.get('plots_folder', 'ft-plots')
        self.plots_folder = os.path.join(self.checkpoint_output_path, self.plots_folder)
        os.makedirs(self.plots_folder, exist_ok=True)
        # Save model .pt files to: 
        self.ft_checkpoints_folder = kwargs.get('ft_checkpoints_folder', 'ft-checkpoints')
        self.ft_checkpoints_folder = os.path.join(self.checkpoint_output_path, self.ft_checkpoints_folder)
        os.makedirs(self.ft_checkpoints_folder, exist_ok=True)
        # Save verbose text / training logs to:
        self.text_logs_folder = kwargs.get('text_logs_folder', 'ft-logs')
        self.text_logs_folder = os.path.join(self.checkpoint_output_path, self.text_logs_folder)
        os.makedirs(self.text_logs_folder, exist_ok=True)

        # Load model and preprocessing - path to Long-CLIP model file:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = longclip.load(checkpoint_input_path, device=self.device)

        self.train_dataset = ArtCaptionDataset(train_or_path, self.caption_field_name, transform=self.preprocess)
        self.val_dataset = ArtCaptionDataset(val_or_path, self.caption_field_name, transform=self.preprocess)

    def trainloop(self):
        """
        Complete training loop. Outputs finetuned checkpoints in designated directory, as well as additional logs
        """
        training_losses = []
        validation_losses = []

        """CONFIG"""
        EPOCHS = self.epochs
        learning_rate = 5e-7
        batch_size = self.batch_size
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(train_dataloader) * EPOCHS
        unfreeze_all = True
        scaler = GradScaler()
        optimizer = AdaBelief(self.model.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.995), weight_decay=1e-3, weight_decouple=False, rectify=True, print_change_log = False)
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.1, anneal_strategy='linear')
        
        min_val_loss = 0 # save only min val loss checkpoints if save_min_loss arg == True
        last_saved = None # track the checkpoint that is saved last
        model = self.model.float()
        print(f"Precision: {model.dtype}")
        print(f'Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}')
        print("== START == \n")
        for epoch in range(EPOCHS):
            gradient_norms = {}
            self._unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
            model.train()
            total_train_loss = 0.0
            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=True)
            for batch_idx, (images, texts) in progress_bar:
                images, texts = images.to(self.device), texts.to(self.device)
                with torch.autocast(device_type=self.device):
                    total_loss = model(images, texts)

                scaler.scale(total_loss).backward()
                # Store gradient norms for plot
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        grad_norm = parameter.grad.norm().item()
                        gradient_norms.setdefault(name, []).append(grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                total_train_loss += total_loss.item()

                progress_bar.set_postfix({'loss': f'{total_train_loss / (batch_idx + 1):.4f}'})
                with open(os.path.join(self.text_logs_folder, "log_details_train.txt"), "a", encoding='utf-8') as f:
                    f.write(f"Epoch {epoch + 1}/{EPOCHS}, Batch: {batch_idx + 1}/{len(train_dataloader)}, Loss: {total_loss.item():.4f}\n")
    
            avg_train_loss = total_train_loss / len(train_dataloader)
            training_losses.append(avg_train_loss)
            self._plot_gradient_norms(gradient_norms, epoch)
            del gradient_norms

            # Validation
            model.eval()    
            print("Running Validation...")
            min_flag = False
            val_total_loss = 0
            with torch.no_grad():
                for images, texts in val_dataloader:
                    images, texts = images.to(self.device), texts.to(self.device)
                    loss = model(images, texts)
                    val_total_loss += loss.item()

            avg_val_loss = val_total_loss / len(val_dataloader)
            validation_losses.append(avg_val_loss)

            if epoch==0:
                min_val_loss = avg_val_loss
                min_flag = True
            else:
                if avg_val_loss <= min_val_loss:
                    min_val_loss = avg_val_loss
                    min_flag = True
            
            if epoch >= 1:
                # Plot losses
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, epoch + 2), training_losses, label='Training Loss')
                plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss Over Epochs')
                plt.legend()
                plt.savefig(f"{self.plots_folder}/loss_plot_epoch_{epoch + 1}.png")
                plt.close()        
            
            
            print(Fore.YELLOW + "======================== STATS =============================")
            print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
            print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)
            
            with open(os.path.join(self.text_logs_folder, "log_training.txt"), "a", encoding='utf-8') as f:
                f.write("======================== STATS =============================\n")
                f.write(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
                f.write("============================================================\n")

            # Check early stopping
            if self.early_stop and not min_flag:
                print("Early stopping due to val loss increase.")
                return last_saved

            # Save model every epoch unless only saving min
            if (self.save_min_loss == False or (self.save_min_loss == True and min_flag == True)):
                save_checkpoint_path = os.path.join(self.ft_checkpoints_folder, f"{epoch+1}.pt")
                torch.save(model.state_dict(), save_checkpoint_path)
                last_saved = f"{epoch+1}.pt"      
                print(Fore.GREEN + f"Checkpoint saved at: {save_checkpoint_path}" + Style.RESET_ALL)
        
        return last_saved

    def _adjust_unfreeze_rate(self, epoch, adjust_after=12, increase_rate=2):
        """
        Adjusts the rate of unfreezing after a certain number of epochs.

        Args:
            epoch: Current epoch number.
            adjust_after: Epoch after which to increase unfreezing rate.
            increase_rate: How many layers to unfreeze per epoch after adjust_after.

        Returns: 
            Number of layers to unfreeze per epoch.
        """
        if epoch < adjust_after:
            return 1  # Initial slower unfreeze rate
        else:
            return increase_rate  # Increased rate after initial pass

    def _unfreeze_layers(self, model, epoch, total_layers=24, unfreeze_all=False):
        if unfreeze_all:
            for param in model.parameters():
                param.requires_grad = True
        else:
            unfreeze_every_n_epochs = self._adjust_unfreeze_rate(epoch)
            layers_to_unfreeze = (epoch // unfreeze_every_n_epochs) % total_layers
            layers_to_unfreeze = min(layers_to_unfreeze, total_layers)
            for i, (name, param) in enumerate(model.named_parameters()):
                if i >= total_layers - layers_to_unfreeze:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def _monitor_gradient_norms(self, gradient_norms, threshold=1e-5):
        alert_messages = []
        for name, norms in gradient_norms.items():
            mean_norm = sum(norms) / len(norms)
            if mean_norm < threshold:  # Vanishing gradient
                alert_messages.append(Fore.RED + f"Vanishing gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
            elif mean_norm > 1000:  # Exploding gradient
                alert_messages.append(Fore.RED + f"Exploding gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        if alert_messages:
            for message in alert_messages:
                print(message)

    def _plot_gradient_norms(self, gradient_norms, epoch, use_log_scale=True):
        plt.figure(figsize=(20, 10))
        
        # Choose a colormap
        cmap = plt.get_cmap('Spectral')
        
        # Sort the layers by the maximum gradient norm value, descending
        sorted_layers = sorted(gradient_norms.items(), key=lambda item: max(item[1]), reverse=True)
        
        # Generate distinct colors from the colormap
        colors = cmap(range(len(sorted_layers)))
        
        for (layer_name, norms), color in zip(sorted_layers, colors):
            plt.plot(norms, label=layer_name, color=color)

        plt.xlabel('Batch')
        plt.ylabel('Gradient Norm')
        #plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
        
        # Adjust legend: position at top right with smaller font size
        plt.legend(loc='upper right', fontsize='small')
        
        # If log scale is requested, change the y-axis to logarithmic
        if use_log_scale:
            plt.yscale('log')
            plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
            plt.savefig(os.path.join(self.plots_folder, f"gradient_norms_epoch_{epoch}_log.png"))
        else:
            plt.savefig(os.path.join(self.plots_folder, f"gradient_norms_epoch_{epoch}.png"))
        
        plt.close()

if __name__ == '__main__':
    parser = FinetuneParser()
    args = parser.parse_args()
    finetuner = finetune(val_path=args.val_path, train_path=args.train_path, caption_field_name=args.cap, checkpoint_output_path=args.checkpoint_out, epochs=args.epochs, save_min_loss=args.save_min, checkpoint_input_path=args.checkpoint_in)
    finetuner.trainloop()