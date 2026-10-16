# 알려진 한계와 미확인 항목

- 미확인: 현재 GPU 드라이버 통신 실패. GPU 사용률·VRAM 한도·GPU 재현은 검증하지 못했다.
- 미확인: 기존 문서의 60M 제한과 상위 CLAUDE.md의 100M 제한이 충돌한다.
  실제 새 연구 실험 전에 적용 기준을 확인한다. 인프라 CPU smoke는 모델 예산 실험이 아니다.
- 미확인: 소급 기록의 parent_id, hypothesis, 실행 command/host/시간/dirty 상태는
  원본에서 확인하지 못하면 null이다. 파일 mtime을 실행 시간으로 추정하지 않는다.
- 소급 metrics.json은 학습 중간 스냅샷일 수도 있다. final.pt 또는 FID 근거 없는 항목을 완료로 확정하지 않는다.
- 미확인: 과거 문서가 가리키는 fixed_cfg2_progressive 원본은 현재 experiments/ 목록에 없다.
  다른 보존 경로와의 lineage 연결은 확인 후 추가한다.
- Development FID와 official validation FID는 reference가 달라 수치를 직접 비교할 수 없다.
  sampling seed 변화는 training seed 재현성이나 reference 불확실성을 뜻하지 않는다.
- 1-NFE 품질 저하/누적 distillation 오차/예산 부족은 기존 리뷰의 가설이며 인과 관계는 미확인이다.
- 자동 sync는 정량 결과 행만 추가한다. 설정/구조/이상 징후의 의미 해석은 Codex가 원본을 읽고 문서를 갱신해야 한다.
- SIGKILL/호스트 재부팅은 finally 기록 불가. start는 PID와 시작시각을 확인하고 failed 후보로만 보고한다.
- JSON 문법 YAML만 자동 생성/파싱한다. 임의의 일반 YAML 문법은 지원하지 않는다.
- 번호 할당은 동일 서버 git lock으로 직렬화된다. 여러 서버에서 동시 할당하는 것은 지원하지 않는다.
- 재현에는 외부 원본 이미지와 체크포인트가 별도로 필요하다. git에는 작은 manifest/metrics만 저장한다.
