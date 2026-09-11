import {defineConfig} from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  timeout: 90_000,
  expect: {timeout: 20_000},
  fullyParallel: false,
  workers: 1,
  reporter: [['list'], ['json', {outputFile: 'test-results/browser-results.json'}]],
  use: {
    baseURL: process.env.AVATAR_LAB_URL ?? 'http://127.0.0.1:8765',
    viewport: {width: 1440, height: 1000},
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
      args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding',
        ...(process.env.CI?['--use-angle=swiftshader','--enable-unsafe-swiftshader']:[]),
        '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
    },
  },
});
