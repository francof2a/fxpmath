<div class="hero">
  <img src="./docs/figs/fxpmath_logotipo.png" alt="fxpmath logo" />
  <p class="lead">
    A Python library for fractional fixed-point (base 2) arithmetic and binary manipulation with NumPy compatibility.
  </p>
  <div class="badges">
    <span class="badge">Develop version: <code>0.4.10-dev0</code></span>
    <span class="badge">Status snapshot: 2026-03-24</span>
    <span class="badge">Pages source: <code>gh-pages</code></span>
  </div>
</div>

<div class="status">
  <strong>Develop branch status (highlights):</strong>
  Packaging modernization (PEP 517/621), improved complex handling, broader bitwise coverage including >64-bit paths, and reshape API alignment with NumPy.
</div>

## What You Can Do With fxpmath

<div class="grid">
  <section class="card">
    <h3>Represent And Scale Precisely</h3>
    <ul>
      <li>Signed/unsigned fixed-point formats</li>
      <li>Arbitrary word and fractional sizes</li>
      <li>Linear scaling with <code>scale</code> and <code>bias</code></li>
      <li>Extended precision workflows</li>
    </ul>
  </section>
  <section class="card">
    <h3>Compute Like A Numeric Type</h3>
    <ul>
      <li>Arithmetic and bitwise operators</li>
      <li>Configurable rounding/overflow behavior</li>
      <li>NumPy interoperability and function dispatch</li>
      <li>Array operations, reshape, reductions and more</li>
    </ul>
  </section>
</div>

## Quick Navigation

<div class="quicklinks">
  <a href="docs/install">Install</a>
  <a href="docs/quick_start">Quick Start</a>
  <a href="docs/config">Configuration</a>
  <a href="docs/generalized_sizing">Generalized Sizing</a>
  <a href="README">Repository README</a>
  <a href="https://github.com/francof2a/fxpmath/blob/develop/changelog.txt">Develop Changelog</a>
</div>

## Recommended GitHub Pages Setup (2026)

1. Keep Pages source explicit (`gh-pages` branch root) or move to a dedicated GitHub Actions Pages deploy workflow when you need tighter build control.
2. Use a custom 404 page and keep navigation links shallow so users can recover quickly from broken URLs.
3. If using a custom domain, verify the domain ownership and keep HTTPS enabled to reduce takeover risk.

<p class="footer-note">
Last content sync target: <code>develop</code> branch state as of 2026-03-24.
</p>
