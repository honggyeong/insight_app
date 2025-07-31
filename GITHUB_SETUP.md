# GitHub 업로드 가이드

## 1. GitHub에서 새 저장소 생성

1. GitHub.com에 로그인
2. 우측 상단의 "+" 버튼 클릭 → "New repository" 선택
3. 저장소 설정:
   - **Repository name**: `daegu-wheelchair-navigation`
   - **Description**: `대구 전동휠체어 길안내 시스템 - AI 기반 실시간 AR 네비게이션`
   - **Visibility**: Public (또는 Private)
   - **README, .gitignore, license 체크 해제** (이미 있으므로)

## 2. 원격 저장소 연결 및 업로드

터미널에서 다음 명령어들을 순서대로 실행하세요:

```bash
# 원격 저장소 추가 (YOUR_USERNAME을 실제 GitHub 사용자명으로 변경)
git remote add origin https://github.com/YOUR_USERNAME/daegu-wheelchair-navigation.git

# 메인 브랜치를 main으로 설정
git branch -M main

# 원격 저장소에 푸시
git push -u origin main
```

## 3. GitHub Pages 설정 (선택사항)

프로젝트를 웹에서 바로 실행할 수 있도록 GitHub Pages를 설정할 수 있습니다:

1. 저장소 설정 → Pages
2. Source를 "Deploy from a branch"로 설정
3. Branch를 "main"으로 선택
4. Save 클릭

## 4. 추가 파일 업로드 (필요시)

```bash
# 파일 수정 후
git add .
git commit -m "Update: [변경사항 설명]"
git push
```

## 5. 이슈 및 프로젝트 관리

GitHub에서 다음 기능들을 활용할 수 있습니다:

- **Issues**: 버그 리포트 및 기능 요청
- **Projects**: 프로젝트 관리 및 작업 추적
- **Wiki**: 추가 문서 작성
- **Releases**: 버전 관리

## 6. 협업 설정

다른 개발자와 협업하려면:

1. **Collaborators** 추가 (Settings → Collaborators)
2. **Branch protection rules** 설정 (Settings → Branches)
3. **Pull Request** 워크플로우 구성

---

**참고**: 실제 GitHub 사용자명으로 `YOUR_USERNAME`을 변경해야 합니다. 