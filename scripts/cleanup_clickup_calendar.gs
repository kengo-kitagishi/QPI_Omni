/**
 * ClickUp から Google カレンダーに同期された残骸イベントの一括削除
 *
 * 背景:
 *   ClickUp の Google Calendar 連携が作った予定（タイトルが "🔄 " で始まり、
 *   説明文に "This task was synced from ClickUp" を含むもの）が大量に残っている。
 *   ClickUp 運用を終了したため、これらは更新されない残骸になる。
 *
 * 使い方:
 *   1. https://script.google.com/ で新しいプロジェクトを作り、このファイルの中身を貼る
 *   2. まず DRY_RUN = true のまま cleanupClickUpEvents を実行し、
 *      「実行ログ」で消える予定の一覧と件数を確認する（この時点では何も消えない）
 *   3. 問題なければ DRY_RUN = false にして、もう一度実行する
 *   4. 「時間切れ」と出たら、同じ関数をもう一度実行する（続きから再開する）
 *
 * 消えないもの:
 *   - 説明文にマーカーが無い予定（自分で入れた「部活」「起きる」などの繰り返し予定はそのまま）
 *   - Notion のタスク（このスクリプトは Google カレンダーしか触らない）
 *
 * やり直したいとき: resetCursor() を実行すると再開位置がリセットされる
 */

var CONFIG = {
  DRY_RUN: true,                                  // false にすると実際に削除する
  START_DATE: '2026-01-01',                       // この日から
  END_DATE: '2028-01-01',                         // この日の前日まで
  MARKER: 'This task was synced from ClickUp',    // 説明文に含まれていたら対象
  CALENDAR_ID: '',                                // 空なら既定のカレンダー
  MAX_RUNTIME_MS: 4.5 * 60 * 1000                 // Apps Script の6分制限に対する余裕
};

var CURSOR_KEY = 'CLICKUP_CLEANUP_CURSOR';

function cleanupClickUpEvents() {
  var props = PropertiesService.getUserProperties();
  var cal = CONFIG.CALENDAR_ID
    ? CalendarApp.getCalendarById(CONFIG.CALENDAR_ID)
    : CalendarApp.getDefaultCalendar();
  if (!cal) throw new Error('カレンダーが見つかりません: ' + CONFIG.CALENDAR_ID);

  var saved = props.getProperty(CURSOR_KEY);
  var day = saved ? new Date(saved) : parseDate_(CONFIG.START_DATE);
  var end = parseDate_(CONFIG.END_DATE);
  var startedAt = Date.now();
  var seen = {};
  var matched = 0, deleted = 0, failed = 0;

  Logger.log('%s: %s 〜 %s',
    CONFIG.DRY_RUN ? '確認のみ（DRY_RUN）' : '削除します',
    fmt_(day), CONFIG.END_DATE);

  while (day.getTime() < end.getTime()) {
    if (Date.now() - startedAt > CONFIG.MAX_RUNTIME_MS) {
      props.setProperty(CURSOR_KEY, day.toISOString());
      Logger.log('--- 時間切れ。%s から再開します。もう一度実行してください ---', fmt_(day));
      summary_(matched, deleted, failed);
      return;
    }

    var next = new Date(day.getTime() + 24 * 60 * 60 * 1000);
    var events = cal.getEvents(day, next);

    for (var i = 0; i < events.length; i++) {
      var ev = events[i];
      var id;
      try {
        id = ev.getId();
      } catch (e) {
        continue;
      }
      if (seen[id]) continue;
      seen[id] = true;

      var desc = '';
      try {
        desc = ev.getDescription() || '';
      } catch (e) {
        continue;
      }
      if (desc.indexOf(CONFIG.MARKER) === -1) continue;

      matched++;
      if (CONFIG.DRY_RUN) {
        Logger.log('[確認] %s  %s', fmt_(ev.getStartTime()), ev.getTitle());
      } else {
        try {
          ev.deleteEvent();
          deleted++;
        } catch (e) {
          failed++;
          Logger.log('[失敗] %s  %s  (%s)', fmt_(ev.getStartTime()), ev.getTitle(), e.message);
        }
      }
    }
    day = next;
  }

  props.deleteProperty(CURSOR_KEY);
  Logger.log('--- 完了 ---');
  summary_(matched, deleted, failed);
}

function resetCursor() {
  PropertiesService.getUserProperties().deleteProperty(CURSOR_KEY);
  Logger.log('再開位置をリセットしました');
}

function summary_(matched, deleted, failed) {
  Logger.log('対象: %s件 / 削除: %s件 / 失敗: %s件', matched, deleted, failed);
  if (CONFIG.DRY_RUN) Logger.log('DRY_RUN = true のため、実際には削除していません');
}

function parseDate_(s) {
  var p = s.split('-');
  return new Date(Number(p[0]), Number(p[1]) - 1, Number(p[2]));
}

function fmt_(d) {
  return Utilities.formatDate(d, Session.getScriptTimeZone(), 'yyyy-MM-dd HH:mm');
}
