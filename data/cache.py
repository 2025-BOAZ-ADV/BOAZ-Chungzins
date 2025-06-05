import os
import torch
import numpy as np
import hashlib
from pathlib import Path

class DataCache:
    """전처리된 데이터를 디스크에 캐싱"""
    
    def __init__(self, cache_dir: str):
        """
        Args:
            cache_dir: 캐시 저장 경로
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, key: str) -> Path:
        """캐시 파일 경로 생성
        
        Args:
            key: 캐시 키
            
        Returns:
            캐시 파일 경로
        """
        # key를 해시로 변환하여 파일 이름으로 사용
        hash_str = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_str}.pt"
        
    def exists(self, key: str) -> bool:
        """캐시 존재 여부 확인"""
        return self._get_cache_path(key).exists()
        
    def save(self, key: str, data: torch.Tensor):
        """데이터를 캐시에 저장"""
        cache_path = self._get_cache_path(key)
        torch.save(data, cache_path)
        
    def load(self, key: str) -> torch.Tensor:
        """캐시에서 데이터 로드"""
        cache_path = self._get_cache_path(key)
        return torch.load(cache_path)
        
    def clear(self):
        """모든 캐시 삭제"""
        for cache_file in self.cache_dir.glob("*.pt"):
            cache_file.unlink()
