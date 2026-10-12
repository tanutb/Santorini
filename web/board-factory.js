import { IslandBoard } from './board.js';

/**
 * One board facade for the 3D renderer and the lightweight Canvas fallback.
 * The 3D module is loaded lazily so the page is usable while it downloads; if WebGL is
 * unavailable, the module fails, or the GPU context is lost, the Canvas board takes over
 * with the same authoritative state.
 */
export function createBoard(canvas, { onPick, onHover, onNotice = () => {}, onMode = () => {} }) {
  let renderer = null, kind = 'loading', latest = null, quality = 'auto';

  function apply() {
    if (!renderer || !latest) return;
    const [game, decor] = latest;
    if (kind === '3d') renderer.setState(game, decor);
    else renderer.setState(game, decor.targets, decor.selected, decor.movable);
  }

  function useCanvas(reason) {
    if (reason) console.warn('Using the lightweight board:', reason);
    renderer?.dispose?.();
    const fresh = canvas.cloneNode(false);
    canvas.replaceWith(fresh);
    canvas = fresh;
    renderer = new IslandBoard(canvas, onPick, onHover);
    kind = 'canvas';
    apply();
    onMode(kind);
  }

  const ready = (async () => {
    try {
      const { ThreeBoard } = await import('./three-board.js');
      renderer = new ThreeBoard(canvas, { onPick, onHover, onNotice, onFailure: () => useCanvas('graphics context lost') });
      renderer.setQuality(quality);
      kind = '3d';
      apply();
      onMode(kind);
    } catch (error) {
      useCanvas(error.message);
    }
  })();

  return {
    ready,
    get kind() { return kind; },
    get canvas() { return canvas; },
    setState(game, decor) { latest = [game, decor]; apply(); },
    rotate(direction) { renderer?.rotate(direction); },
    resetView() { renderer?.resetView(); },
    zoomBy(delta) { renderer?.zoomBy?.(delta); },
    setQuality(value) { quality = value; renderer?.setQuality?.(value); },
    replayLast() { renderer?.replayLast?.(); },
    cellScreenPosition(cell) { return renderer?.cellScreenPosition?.(cell) ?? null; },
    cellAt(x, y) { return renderer?.pick?.(x, y) ?? -1; },
    get diagnostics() { return renderer?.diagnostics ?? { renderer: kind }; },
  };
}
