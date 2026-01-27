"""
PDDL to DAG Generator with Local LLM (Ollama)
FastDownward 출력을 DAG로 변환하고 여러 subtask DAG를 통합하는 모듈
필요시 TaskManager에 통합 가능
"""

import re
import json
import time
import requests
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import os


# LLM 설정
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_RETRY_DELAY = 2
MAX_RETRIES = 3


class LLMError(Exception):
    """LLM 관련 에러"""
    pass


class LLMHandler:
    """Handles interactions with Language Models (LLMs) via Ollama."""
    
    def __init__(
        self,
        api_key_file: str = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "gpt-oss:20b"
    ):
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        
    def setup_api(self, api_key_file: str) -> None:
        return
    
    def _normalize_messages(self, prompt):
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt
    
    def query_model(
        self, 
        prompt, 
        gpt_version: str = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        stop: Optional[List[str]] = None,
        logprobs: Optional[int] = 1,
        frequency_penalty: float = 0
    ) -> Tuple[dict, str]:
        messages = self._normalize_messages(prompt)
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if stop:
            payload["options"]["stop"] = stop
        
        retry_delay = DEFAULT_RETRY_DELAY
        last_err = None
        
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json=payload,
                    timeout=300
                )
                r.raise_for_status()
                data = r.json()
                text = data.get("message", {}).get("content", "")
                return data, text.strip()
            except Exception as e:
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise LLMError(f"Ollama API Error after all retries: {str(last_err)}")


class PDDLAction:
    """PDDL Action 표현 클래스"""
    
    def __init__(self, action_id: str, action_type: str, params: List[str], subtask_id: str = None):
        self.id = action_id
        self.action_type = action_type
        self.params = params
        self.subtask_id = subtask_id
        
    def __repr__(self):
        return f"{self.id}: {self.action_type} {' '.join(self.params)}"
    
    def to_dict(self):
        return {
            'id': self.id,
            'action_type': self.action_type,
            'params': self.params,
            'subtask_id': self.subtask_id
        }


class DAG:
    """Directed Acyclic Graph 클래스"""
    
    def __init__(self, subtask_id: str = None):
        self.subtask_id = subtask_id
        self.nodes: Dict[str, PDDLAction] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)
        self.reverse_edges: Dict[str, List[str]] = defaultdict(list)
        self.layers: List[List[str]] = []
        
    def add_node(self, action: PDDLAction):
        """노드 추가"""
        self.nodes[action.id] = action
        
    def add_edge(self, source_id: str, target_id: str):
        """엣지 추가"""
        if source_id not in self.edges[target_id]:
            self.edges[target_id].append(source_id)
        if target_id not in self.reverse_edges[source_id]:
            self.reverse_edges[source_id].append(target_id)
            
    def compute_layers(self):
        """위상 정렬을 통한 레이어 계산"""
        in_degree = {node_id: len(deps) for node_id, deps in self.edges.items()}
        
        for node_id in self.nodes:
            if node_id not in in_degree:
                in_degree[node_id] = 0
                
        self.layers = []
        processed = set()
        
        while len(processed) < len(self.nodes):
            current_layer = []
            for node_id in self.nodes:
                if node_id in processed:
                    continue
                    
                deps = self.edges.get(node_id, [])
                if all(dep in processed for dep in deps):
                    current_layer.append(node_id)
                    
            if not current_layer:
                remaining = set(self.nodes.keys()) - processed
                print(f"Warning: Circular dependency detected in nodes: {remaining}")
                break
                
            self.layers.append(current_layer)
            processed.update(current_layer)
            
    def get_statistics(self) -> Dict:
        """DAG 정보"""
        return {
            'total_actions': len(self.nodes),
            'total_edges': sum(len(deps) for deps in self.edges.values()),
            'total_layers': len(self.layers),
            'max_parallelism': max(len(layer) for layer in self.layers) if self.layers else 0,
            'subtask_id': self.subtask_id
        }
        
    def to_dict(self) -> Dict:
        """JSON 직렬화를 위한 딕셔너리 변환"""
        return {
            'subtask_id': self.subtask_id,
            'nodes': [action.to_dict() for action in self.nodes.values()],
            'edges': [
                {'source': src, 'target': tgt}
                for tgt, srcs in self.edges.items()
                for src in srcs
            ],
            'layers': self.layers,
            'statistics': self.get_statistics()
        }
        
    def visualize(self, filename: str = None, title: str = None):
        """통합된 모든 액션을 PNG에 표시"""
        G = nx.DiGraph()
        
        for node_id, action in self.nodes.items():
            # subtask 정보를 라벨에 포함하여 중복 액션 구분
            label = f"[{action.subtask_id}]\n{node_id}: {action.action_type}"
            G.add_node(node_id, label=label)
            
        for target, sources in self.edges.items():
            for source in sources:
                G.add_edge(source, target)
                
        # 레이어 기반 배치 설정
        pos = {}
        for layer_idx, layer in enumerate(self.layers):
            for node_idx, node_id in enumerate(layer):
                pos[node_id] = (node_idx * 3, -layer_idx * 2)
                
        plt.figure(figsize=(15, 10))
        
        nx.draw(G, pos, 
                with_labels=True,
                labels=nx.get_node_attributes(G, 'label'),
                node_size=5000,
                node_color='lightblue',
                font_size=8,
                arrows=True)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

class PDDLParser:
    """FastDownward 출력 파싱"""
    
    @staticmethod
    def parse_file(filepath: str, subtask_id: str = None) -> List[PDDLAction]:
        """PDDL 파일 파싱"""
        with open(filepath, 'r') as f:
            content = f.read()
        return PDDLParser.parse_text(content, subtask_id)
    
    @staticmethod
    def parse_text(text: str, subtask_id: str = None) -> List[PDDLAction]:
        """PDDL 텍스트 파싱"""
        actions = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = re.match(r'^(\d+\.\d+):\s+(\w+)\s+(.+)$', line)
            if match:
                action_id, action_type, params_str = match.groups()
                params = params_str.split()
                actions.append(PDDLAction(action_id, action_type, params, subtask_id))
            else:
                print(f"Warning: Could not parse line: {line}")
                
        return actions


class LLMDependencyAnalyzer:
    """LLM 사용 작업 간 의존성 분석"""
    
    def __init__(self, llm_handler: LLMHandler):
        self.llm = llm_handler
        
    def analyze_dependencies(self, actions: List[PDDLAction]) -> Dict[str, List[str]]:
        """
        LLM을 사용하여 작업 간 의존성 분석
        Returns: {target_id: [source_ids]}
        """
        print("Analyzing dependencies using LLM...")
        
        actions_str = "\n".join([str(action) for action in actions])
        
        prompt = f"""You are a task dependency analyzer for robotic planning systems.

Given the following PDDL actions, analyze the dependencies between them and output ONLY a valid JSON object.

Actions:
{actions_str}

Rules for dependency analysis:
1. OpenObject must be completed before StoreObject or PutObject on that container
2. StoreObject/PutObject must be completed before CloseObject on that container
3. PickupObject must be completed before PutObject for that object
4. If an action modifies an object's state, subsequent actions on that object depend on it

Output format (JSON only, no explanations):
{{
  "dependencies": {{
    "action_id": ["prerequisite_action_id1", "prerequisite_action_id2"],
    ...
  }}
}}

Example:
{{
  "dependencies": {{
    "2.0": ["0.0"],
    "5.0": ["2.0", "3.0"]
  }}
}}

Now analyze the given actions and output ONLY the JSON:"""

        try:
            _, response = self.llm.query_model(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.3
            )
            
            dependencies = self._parse_llm_response(response, actions)
            
            print(f"LLM analysis complete. Found {sum(len(v) for v in dependencies.values())} dependencies.")
            return dependencies
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            print("Falling back to rule-based analysis...")
            return self._fallback_rule_based_analysis(actions)
    
    def _parse_llm_response(self, response: str, actions: List[PDDLAction]) -> Dict[str, List[str]]:
        """LLM 응답을 파싱하여 의존성 딕셔너리 추출"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")
            
            data = json.loads(json_str)
            dependencies = data.get('dependencies', {})
            
            action_ids = {action.id for action in actions}
            validated_deps = defaultdict(list)
            
            for target_id, source_ids in dependencies.items():
                if target_id in action_ids:
                    for source_id in source_ids:
                        if source_id in action_ids:
                            validated_deps[target_id].append(source_id)
            
            return dict(validated_deps)
            
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}\nResponse: {response}")
    
    def _fallback_rule_based_analysis(self, actions: List[PDDLAction]) -> Dict[str, List[str]]:
        """LLM 실패 시 fallback: 규칙 기반 의존성 분석"""
        dependencies = defaultdict(list)
        object_state = {}
        container_state = {}
        
        for action in actions:
            action_type = action.action_type
            
            if action_type == 'OpenObject':
                container = action.params[0]
                container_state[container] = {
                    'opened_by': action.id,
                    'stored_items': []
                }
                
            elif action_type == 'StoreObject':
                obj, source, container = action.params
                if container in container_state and 'opened_by' in container_state[container]:
                    dependencies[action.id].append(container_state[container]['opened_by'])
                    container_state[container]['stored_items'].append(action.id)
                object_state[obj] = action.id
                
            elif action_type == 'CloseObject':
                container = action.params[0]
                if container in container_state:
                    for store_action_id in container_state[container].get('stored_items', []):
                        dependencies[action.id].append(store_action_id)
                        
            elif action_type == 'PickupObject':
                obj = action.params[0]
                if obj in object_state:
                    dependencies[action.id].append(object_state[obj])
                object_state[obj] = action.id
                
            elif action_type == 'PutObject':
                obj, location = action.params
                if obj in object_state:
                    dependencies[action.id].append(object_state[obj])
                object_state[obj] = action.id
                
        return dependencies


class DAGGenerator:
    """LLM 기반 DAG 생성 클래스"""
    
    def __init__(self, llm_handler: LLMHandler = None):
        self.llm = llm_handler or LLMHandler()
        self.dependency_analyzer = LLMDependencyAnalyzer(self.llm)
    
    def create_dag_from_actions(self, actions: List[PDDLAction], subtask_id: str = None) -> DAG:
        """PDDLAction 리스트로부터 DAG 생성 (LLM 사용)"""
        dag = DAG(subtask_id)
        
        for action in actions:
            dag.add_node(action)
            
        dependencies = self.dependency_analyzer.analyze_dependencies(actions)
        
        for target_id, source_ids in dependencies.items():
            for source_id in source_ids:
                dag.add_edge(source_id, target_id)
                
        dag.compute_layers()
        
        return dag
    
    def create_dag_from_file(self, filepath: str, subtask_id: str = None) -> DAG:
        """파일로부터 DAG 생성"""
        actions = PDDLParser.parse_file(filepath, subtask_id)
        return self.create_dag_from_actions(actions, subtask_id)
    
    def create_dag_from_text(self, text: str, subtask_id: str = None) -> DAG:
        """텍스트로부터 DAG 생성"""
        actions = PDDLParser.parse_text(text, subtask_id)
        return self.create_dag_from_actions(actions, subtask_id)


class DAGMerger:
    """여러 subtask의 DAG를 하나로 통합"""
    
    def __init__(self, llm_handler: LLMHandler = None):
        self.llm = llm_handler or LLMHandler()
    
    def merge_dags(self, dags: List[DAG], merge_strategy: str = 'llm') -> DAG:
        """여러 DAG를 하나로 통합"""
        if merge_strategy == 'llm':
            return self._merge_with_llm(dags)
        elif merge_strategy == 'sequential':
            return self._merge_sequential(dags)
        elif merge_strategy == 'parallel':
            return self._merge_parallel(dags)
        elif merge_strategy == 'dependency':
            return self._merge_with_dependencies(dags)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
    
    def _merge_with_llm(self, dags: List[DAG]) -> DAG:
        """
        각 Subtask마다 고유한 Root 노드를 생성하고,
        통합 DAG는 이 Root들 사이의 선후 관계만 정의합니다.
        """
        print("Merging DAGs: One Root per Subtask Strategy")
        
        merged = DAG(subtask_id='merged_hierarchical')
        
        # 1. 전체 시스템의 시작점
        global_start = PDDLAction(action_id="GLOBAL_START", action_type="System", params=["Start"])
        merged.add_node(global_start)

        subtask_root_ids = {} # {subtask_id: root_node_id}
        subtask_exit_nodes = {} # {subtask_id: [last_nodes]}

        for dag in dags:
            # 2. 각 Subtask를 대표하는 명시적 'Subtask Root' 생성
            s_root_id = f"{dag.subtask_id}_ROOT"
            s_root_action = PDDLAction(action_id=s_root_id, action_type="Subtask_Start", params=[dag.subtask_id])
            merged.add_node(s_root_action)
            subtask_root_ids[dag.subtask_id] = s_root_id
            
            # 기본적으로 모든 Subtask Root는 글로벌 스타트에 연결 (병렬 대기)
            merged.add_edge("GLOBAL_START", s_root_id)

            # 3. Subtask 내부 액션 및 의존성 복사
            for node_id, action in dag.nodes.items():
                u_id = f"{dag.subtask_id}_{node_id}"
                merged.add_node(PDDLAction(u_id, action.action_type, action.params, dag.subtask_id))
            
            for target, sources in dag.edges.items():
                for src in sources:
                    merged.add_edge(f"{dag.subtask_id}_{src}", f"{dag.subtask_id}_{target}")

            # 4. Subtask Root를 해당 Subtask의 첫 번째 액션들에 연결
            if dag.layers:
                for first_node in dag.layers[0]:
                    merged.add_edge(s_root_id, f"{dag.subtask_id}_{first_node}")
                
                # 의존성 연결을 위해 마지막 노드들 보관
                subtask_exit_nodes[dag.subtask_id] = [f"{dag.subtask_id}_{nid}" for nid in dag.layers[-1]]

        # 5. LLM에게 Subtask Root 간의 의존성 질문
        prompt = f"""Analyze dependencies between subtasks.
Current Subtasks: {list(subtask_root_ids.keys())}

Rules:
1. If Subtask B requires results from Subtask A, create a dependency.
2. Dependencies must only be between Subtask names.

Output JSON:
{{ "dependencies": [ {{ "from": "subtask_1", "to": "subtask_2" }} ] }}"""

        try:
            _, response = self.llm.query_model(prompt=prompt, temperature=0.2)
            import json
            deps = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group(0)).get('dependencies', [])

            # 6. Root 간 의존성 부여 (A의 종료 노드들 -> B의 Root 노드)
            for dep in deps:
                from_sub = dep.get('from')
                to_sub = dep.get('to')

                if from_sub in subtask_exit_nodes and to_sub in subtask_root_ids:
                    target_root = subtask_root_ids[to_sub]
                    for exit_node in subtask_exit_nodes[from_sub]:
                        merged.add_edge(exit_node, target_root)
                        print(f"  [Link] {from_sub} Complete -> {target_root} (Start)")
                            
        except Exception as e:
            print(f"LLM Merge failed: {e}")

        merged.compute_layers()
        return merged
    
    def _merge_sequential(self, dags: List[DAG]) -> DAG:
        """순차적 통합"""
        merged = DAG(subtask_id='merged_sequential')
        previous_dag_last_nodes = []
        
        for dag in dags:
            for node_id, action in dag.nodes.items():
                merged.add_node(action)
            for target, sources in dag.edges.items():
                for source in sources:
                    merged.add_edge(source, target)
            
            if previous_dag_last_nodes and dag.layers:
                first_layer_nodes = dag.layers[0]
                for prev_node in previous_dag_last_nodes:
                    for curr_node in first_layer_nodes:
                        merged.add_edge(prev_node, curr_node)
            
            if dag.layers:
                previous_dag_last_nodes = dag.layers[-1]
                
        merged.compute_layers()
        return merged
    
    def _merge_parallel(self, dags: List[DAG]) -> DAG:
        """병렬 통합"""
        merged = DAG(subtask_id='merged_parallel')
        
        for dag in dags:
            for node_id, action in dag.nodes.items():
                merged.add_node(action)
            for target, sources in dag.edges.items():
                for source in sources:
                    merged.add_edge(source, target)
                    
        merged.compute_layers()
        return merged
    
    def _merge_with_dependencies(self, dags: List[DAG]) -> DAG:
        """규칙 기반 의존성 통합"""
        merged = DAG(subtask_id='merged_dependency')
        
        for dag in dags:
            for node_id, action in dag.nodes.items():
                merged.add_node(action)
            for target, sources in dag.edges.items():
                for source in sources:
                    merged.add_edge(source, target)
        
        object_producers = {}
        
        for dag in dags:
            for action in dag.nodes.values():
                if action.action_type in ['StoreObject', 'PutObject']:
                    obj = action.params[0]
                    object_producers[obj] = (dag.subtask_id, action.id)
        
        for dag in dags:
            for action in dag.nodes.values():
                if action.action_type in ['PickupObject', 'StoreObject']:
                    obj = action.params[0]
                    if obj in object_producers:
                        producer_subtask, producer_action = object_producers[obj]
                        if producer_subtask != dag.subtask_id:
                            merged.add_edge(producer_action, action.id)
                            
        merged.compute_layers()
        return merged


class DAGManager:
    """DAG 관리 클래스 (TaskManager와 통합 가능)"""
    
    def __init__(self, llm_handler: LLMHandler = None, log_folder: str = None):
        """
        Args:
            llm_handler: LLM 핸들러 (None이면 새로 생성)
            log_folder: DAG 결과를 저장할 폴더 경로
        """
        self.llm = llm_handler or LLMHandler()
        self.generator = DAGGenerator(self.llm)
        self.merger = DAGMerger(self.llm)
        self.log_folder = log_folder
        
        self.subtask_dags: List[DAG] = []
        self.merged_dag: Optional[DAG] = None
        
    def set_log_folder(self, log_folder: str):
        """로그 폴더 설정"""
        self.log_folder = log_folder
        
    def generate_subtask_dags(self, plan_files: List[str]) -> List[DAG]:
        """
        여러 subtask plan 파일들로부터 각각의 DAG 생성
        
        Args:
            plan_files: plan 파일 경로 리스트 (예: ["subtask1_plan.txt", ...])
            
        Returns:
            생성된 DAG 리스트
        """
        self.subtask_dags = []
        
        for i, plan_file in enumerate(plan_files):
            try:
                subtask_id = f"subtask_{i+1}"
                print(f"\n=== Generating DAG for {subtask_id} ===")
                
                dag = self.generator.create_dag_from_file(plan_file, subtask_id)
                self.subtask_dags.append(dag)
                
                print(f"✓ DAG generated: {dag.get_statistics()}")
                
                # 개별 DAG 저장
                if self.log_folder:
                    self._save_subtask_dag(dag, i)
                    
            except Exception as e:
                print(f"Error generating DAG for {plan_file}: {e}")
                
        return self.subtask_dags
    
    def merge_all_subtasks(self, merge_strategy: str = 'llm') -> DAG:
        """
        모든 subtask DAG를 하나로 통합
        
        Args:
            merge_strategy: 'llm', 'sequential', 'parallel', 'dependency'
            
        Returns:
            통합된 DAG
        """
        if not self.subtask_dags:
            raise ValueError("No subtask DAGs to merge. Run generate_subtask_dags first.")
            
        print(f"\n=== Merging {len(self.subtask_dags)} DAGs using '{merge_strategy}' strategy ===")
        
        self.merged_dag = self.merger.merge_dags(self.subtask_dags, merge_strategy)
        
        print(f"✓ Merged DAG: {self.merged_dag.get_statistics()}")
        
        # 통합 DAG 저장
        if self.log_folder:
            self._save_merged_dag(merge_strategy)
            
        return self.merged_dag
    
    def _save_subtask_dag(self, dag: DAG, index: int):
        """개별 subtask DAG 저장"""
        dag_folder = os.path.join(self.log_folder, "dags")
        os.makedirs(dag_folder, exist_ok=True)
        
        # JSON 저장
        json_path = os.path.join(dag_folder, f"subtask_{index+1}_dag.json")
        with open(json_path, 'w') as f:
            json.dump(dag.to_dict(), f, indent=2)
        print(f"  Saved DAG JSON: {json_path}")
        
        # 시각화 저장
        viz_path = os.path.join(dag_folder, f"subtask_{index+1}_dag.png")
        dag.visualize(viz_path, title=f"Subtask {index+1} DAG")
        
    def _save_merged_dag(self, strategy: str):
        """통합 DAG 저장"""
        dag_folder = os.path.join(self.log_folder, "dags")
        os.makedirs(dag_folder, exist_ok=True)
        
        # JSON 저장
        json_path = os.path.join(dag_folder, f"merged_dag_{strategy}.json")
        with open(json_path, 'w') as f:
            json.dump(self.merged_dag.to_dict(), f, indent=2)
        print(f"  Saved merged DAG JSON: {json_path}")
        
        # 시각화 저장
        viz_path = os.path.join(dag_folder, f"merged_dag_{strategy}.png")
        self.merged_dag.visualize(viz_path, title=f"Merged DAG ({strategy})")
        
        # 관련 정보 저장
        stats_path = os.path.join(dag_folder, f"merged_dag_{strategy}_stats.txt")
        with open(stats_path, 'w') as f:
            stats = self.merged_dag.get_statistics()
            f.write("=== Merged DAG Statistics ===\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n=== Layer Information ===\n\n")
            for i, layer in enumerate(self.merged_dag.layers):
                f.write(f"Layer {i+1} (parallelism: {len(layer)}):\n")
                for node_id in layer:
                    action = self.merged_dag.nodes[node_id]
                    f.write(f"  - {action}\n")
                f.write("\n")
        print(f"  Saved statistics: {stats_path}")
    
    def save_all_results(self):
        """모든 DAG 결과를 저장 및 요약 리포트 생성"""
        if not self.log_folder:
            print("Warning: No log folder set. Cannot save results.")
            return
            
        # dags 폴더가 없으면 생성
        dag_folder = os.path.join(self.log_folder, "dags")
        os.makedirs(dag_folder, exist_ok=True)
        
        # Summary 파일 생성
        summary_path = os.path.join(dag_folder, "dag_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("==========================================\n")
            f.write("       DAG Generation Summary Report      \n")
            f.write("==========================================\n\n")
            f.write(f"Total Subtasks Processed: {len(self.subtask_dags)}\n\n")
            
            # 각 Subtask 기록
            for i, dag in enumerate(self.subtask_dags):
                f.write(f"--- Subtask {i+1} ({dag.subtask_id}) ---\n")
                stats = dag.get_statistics()
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # 통합 DAG 기록
            if self.merged_dag:
                f.write("--- Final Merged DAG ---\n")
                stats = self.merged_dag.get_statistics()
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                
                f.write("\n[Critical Path / Layer Structure]\n")
                for idx, layer in enumerate(self.merged_dag.layers):
                    f.write(f"  Layer {idx}: {', '.join(layer)}\n")
            
            f.write("\n==========================================\n")
            f.write(f"Report Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"✓ All DAG results and summary saved to: {dag_folder}")

def run_lamma_p_dag_pipeline():
    # 1. 초기화
    llm_handler = LLMHandler(ollama_model="gpt-oss:20b")
    
    # 결과물을 저장할 로그 폴더 설정
    log_dir = "./logs"
    dag_manager = DAGManager(llm_handler=llm_handler, log_folder=log_dir)

    # 2. 고수준 명령어로부터 분해된 Subtask 플랜들 (FastDownward 결과 가정)
    sample_plans = {
        "subtask_1.txt": """
            0.0: OpenObject fridge1
            1.0: OpenObject fridge2
            2.0: StoreObject apple table1 fridge1
            3.0: StoreObject banana table1 fridge1
            4.0: StoreObject lettuce table1 fridge2
            5.0: CloseObject fridge1
            6.0: CloseObject fridge2
        """,
        "subtask_2.txt": """
            0.0: PickupObject apple table1
            1.0: PickupObject banana table1
            2.0: SliceObject apple
            3.0: PutObject apple table1
            4.0: PutObject banana table1
        """
    }

    # 임시 테스트 파일 생성
    plan_files = []
    os.makedirs("./subtask_pddl_plans", exist_ok=True)
    for filename, content in sample_plans.items():
        path = os.path.join("./subtask_pddl_plans", filename)
        with open(path, "w") as f:
            f.write(content.strip())
        plan_files.append(path)

    print("\n>>> [STEP 1] Subtask별 DAG 생성 시작")
    dag_manager.generate_subtask_dags(plan_files)

    print("\n>>> [STEP 2] 개별 DAG 통합 및 의존성 분석")
    merged_dag = dag_manager.merge_all_subtasks(merge_strategy='llm')

    print("\n>>> [STEP 3] 통합 DAG 레이어 분석 (병렬 실행 그룹화)")
    for i, layer in enumerate(merged_dag.layers):
        actions = [f"({node_id})" for node_id in layer]
        print(f"  Layer {i}: {', '.join(actions)} 실행 가능")

    print("\n>>> [STEP 4] 결과물 저장 및 리포트 생성")
    dag_manager.save_all_results()

    print("\n[완료] 모든 DAG 생성 및 통합이 마무리되었습니다.")
    print(f"결과 확인 경로: {os.path.abspath(log_dir)}")

if __name__ == "__main__":
    run_lamma_p_dag_pipeline()
