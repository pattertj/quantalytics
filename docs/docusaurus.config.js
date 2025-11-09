// @ts-check

const config = {
  title: 'Quantalytics',
  tagline: 'Fast, modern quantitative analysis in Python',
  favicon: 'img/favicon.ico',

  url: 'https://pattertj.github.io',
  baseUrl: '/quantalytics/',

  organizationName: 'quantalytics',
  projectName: 'quantalytics',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */ ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/quantalytics/quantalytics/tree/main/docs/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
};

module.exports = config;
