import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';

export default function Home() {
  return (
    <Layout
      title="Quantalytics"
      description="Fast, modern quantitative analysis in Python"
    >
      <header className="hero hero--primary">
        <div className="container">
          <h1 className="hero__title">Quantalytics</h1>
          <p className="hero__subtitle">
            Compute performance metrics, plot beautiful charts, and ship reports — all from Python.
          </p>
          <div className={styles.buttons}>
            <Link className="button button--secondary button--lg" to="/docs/intro">
              Get Started
            </Link>
          </div>
        </div>
      </header>
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--4">
                <h3>Metrics Toolkit</h3>
                <p>Sharpe, Sortino, Calmar, drawdowns, and rolling stats with a single call.</p>
              </div>
              <div className="col col--4">
                <h3>Interactive Visuals</h3>
                <p>Plotly-powered charts with defaults tailored to quant research.</p>
              </div>
              <div className="col col--4">
                <h3>Production Reports</h3>
                <p>Generate responsive HTML tear sheets ready to export and share.</p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
