from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


class FasterRCNNModel:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.optimizer = None
        self.model = self.create_model().to(self.device)
        logger.info("Model Architecture initialized....")

    def create_model(self):
        try:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features , self.num_classes)
            return model
        except Exception as e:
            logger.error("Failed to create the model: ", e)
            raise CustomException("Failed to create the model: ",e)
    
    def compile(self, lr=1e-4):
        try:
            self.optimizer = Adam(self.model.parameters(), lr=lr)
            logger.info("Model compiled sucesfully")
        except Exception as e:
            logger.error(f"Failed to compile the model {e}")
            raise CustomException("Failed to compile the model: ", e)
        
    
    def train(self, train_loader, num_epoch=10):
        try:
            self.model.train()
            for epoch in range(1, num_epoch+1):
                total_loss = 0
                logger.info(f"Epoch {epoch} started....")
                for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
                    images = [img.to(self.device) for img in images]
                    targets = [{key:val.to(self.device) for key, val in target.items()} for target in targets]

                    loss_dict = self.model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                logger.info(f"Epoch {epoch} completed with train loss: {total_loss}")
                
        except Exception as e:
            logger.error(f"Failed to train the model {e}")
            raise CustomException("Failed to train the model: ", e)