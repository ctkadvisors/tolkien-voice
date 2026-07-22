#!/usr/bin/env bash
set -Eeuo pipefail

readonly expected_revision=3428dfd95309a7f3c84fd93259ded0f810d1ff91
actual_revision=$(git -C vendor/personaplex rev-parse HEAD)
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "PersonaPlex submodule is $actual_revision; expected $expected_revision" >&2
  exit 65
fi

rsync -a --delete \
  --exclude '.env.local' \
  --exclude 'node_modules/' \
  --exclude 'dist/' \
  --exclude 'e2e/' \
  --exclude 'package.json' \
  --exclude 'package-lock.json' \
  --exclude 'index.html' \
  --exclude 'vite.config.ts' \
  --exclude 'vitest.config.ts' \
  --exclude 'playwright.config.ts' \
  --exclude 'tailwind.config.js' \
  --exclude 'tsconfig.json' \
  --exclude 'UPSTREAM.md' \
  --exclude 'src/app.tsx' \
  --exclude 'src/Root.tsx' \
  --exclude 'src/api.ts' \
  --exclude 'src/types.ts' \
  --exclude 'src/test-setup.ts' \
  --exclude 'src/index.css' \
  --exclude 'src/pages/CharacterSelect.tsx' \
  --exclude 'src/pages/CharacterSelect.test.tsx' \
  --exclude 'src/pages/ConversationPage.tsx' \
  --exclude 'src/pages/ConversationPage.test.tsx' \
  --exclude 'src/pages/Queue/' \
  --exclude 'src/pages/Conversation/Conversation.tsx' \
  --exclude 'src/pages/Conversation/MediaContext.ts' \
  --exclude 'src/pages/Conversation/getMimeType.ts' \
  --exclude 'src/pages/Conversation/components/ModelParams/' \
  --exclude 'src/pages/Conversation/hooks/useSocket.ts' \
  --exclude 'src/pages/Conversation/hooks/useServerAudio.ts' \
  --exclude 'src/pages/Conversation/hooks/useUserAudio.ts' \
  --exclude 'src/pages/Conversation/hooks/useModelParams.ts' \
  --exclude 'src/pages/Conversation/components/UserAudio/UserAudio.tsx' \
  vendor/personaplex/client/ web/
