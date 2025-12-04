from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from .types import CrawlResult
import json
import logging

logger = logging.getLogger(__name__)


class JSONStorage:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(exist_ok=True)

    async def save(self, task_name: str, results: List[CrawlResult]) -> None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{task_name}_clean_{timestamp}.json"
        filepath = self.output_dir / filename

        serializable_results = [self._serialize_result(result) for result in results]

        data = {
            'task_name': task_name,
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'successful_results': len([r for r in results if r.content]),
            'results': serializable_results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _serialize_result(self, result: CrawlResult) -> Dict[str, Any]:
        return {
            'url': result.url,
            'original_url': result.original_url,
            'content': result.content,
            'timestamp': result.timestamp,
            'new_tab': result.new_tab,
            'strategy_used': result.strategy_used.value if result.strategy_used else None,
            'button_info': {
                'text': result.button_info.get('text', ''),
                'selector': result.button_info.get('selector', ''),
                'href': result.button_info.get('href')
            }
        }


class StorageFactory:

    @staticmethod
    def create_storage(storage_type: str, config: Dict[str, Any]):
        if storage_type == 'json':
            return JSONStorage(config)
        else:
            raise ValueError(f"不支持的存储类型: {storage_type}")
