/** Short synthesized cues. Off by default; nothing plays until the player turns sound on. */
const NOTES = {
  select: [520], move: [392, 587], place: [440, 660], build: [294, 440], dome: [440, 660, 880],
  win: [392, 494, 587, 784], lose: [392, 330, 262], turn: [660, 880], error: [196],
};

export class Feedback {
  constructor() { this.enabled = false; this.context = null; }

  async unlock() {
    if (!this.enabled) return;
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) throw new Error('Audio is unavailable in this browser.');
    this.context ??= new Context();
    if (this.context.state === 'suspended') await this.context.resume();
  }

  play(kind) {
    if (!this.enabled || this.context?.state !== 'running') return;
    const now = this.context.currentTime;
    (NOTES[kind] || NOTES.select).forEach((frequency, index) => {
      const oscillator = this.context.createOscillator(), gain = this.context.createGain(), start = now + index * .08;
      oscillator.type = kind === 'error' ? 'triangle' : 'sine';
      oscillator.frequency.value = frequency;
      oscillator.connect(gain);
      gain.connect(this.context.destination);
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(.05, start + .012);
      gain.gain.exponentialRampToValueAtTime(.001, start + .2);
      oscillator.start(start);
      oscillator.stop(start + .22);
      oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
    });
  }
}
