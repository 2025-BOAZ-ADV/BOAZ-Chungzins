import wandb
from typing import Dict, Any

class WandbLogger:
    def __init__(self, project_name: str, config: Dict[str, Any] = None, entity: str = None):
        """Weights & Biases 로거 초기화
        
        Args:
            project_name: wandb 프로젝트 이름
            config: 설정값 딕셔너리
            entity: wandb entity(팀/계정명)
        """
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            config=config
        )
    
    def log(self, metrics: Dict[str, float]):
        """메트릭 로깅
        
        Args:
            metrics: 로깅할 메트릭 딕셔너리
        """
        wandb.log(metrics)
    
    def log_model(self, model_path: str, aliases: list = None):
        """모델 아티팩트 로깅
        
        Args:
            model_path: 모델 체크포인트 경로
            aliases: 모델 별칭 리스트
        """
        artifact = wandb.Artifact(
            name=f'model-{wandb.run.id}',
            type='model',
            metadata=wandb.config.as_dict()
        )
        artifact.add_file(model_path)
        
        if aliases:
            self.run.log_artifact(artifact, aliases=aliases)
        else:
            self.run.log_artifact(artifact)
    
    def finish(self):
        """wandb 실행 종료"""
        wandb.finish()
