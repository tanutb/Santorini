import test from 'node:test';
import assert from 'node:assert/strict';
import { Tweens, easing } from '../web/tween.js';

test('tweens advance with the clock and stop requesting frames when done', () => {
  let now = 0, seen = [], done = 0;
  const tweens = new Tweens(() => now);
  tweens.add({ duration: 100, ease: easing.linear, update: value => seen.push(value), done: () => done++ });
  now = 50; assert.equal(tweens.step(now), true);
  now = 150; assert.equal(tweens.step(now), false);
  assert.deepEqual(seen, [.5, 1]);
  assert.equal(done, 1);
});

test('delayed tweens wait, cancel works, and finish jumps to the end', () => {
  let now = 0, last = null;
  const tweens = new Tweens(() => now);
  tweens.add({ delay: 100, duration: 100, ease: easing.linear, update: value => { last = value; } });
  const cancel = tweens.add({ duration: 100, update: () => assert.fail('cancelled tween ran') });
  cancel();
  tweens.step(50);
  assert.equal(last, null);
  tweens.finish();
  assert.equal(last, 1);
  assert.equal(tweens.busy, false);
});

test('easing curves start at 0 and end at 1', () => {
  for (const [name, curve] of Object.entries(easing)) {
    assert.ok(Math.abs(curve(0)) < 1e-9, name);
    assert.ok(Math.abs(curve(1) - 1) < 1e-9, name);
  }
});
