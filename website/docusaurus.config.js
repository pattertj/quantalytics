// @ts-check

const config = {
  title: 'Quantalytics',
  tagline: 'Modern quantitative analytics toolkit',
  url: 'https://docs.quantalytics.dev',
  baseUrl: '/',
  favicon: 'https://fav.farm/📈',
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
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/example/quantalytics/edit/main/website/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  themeConfig: {
    image: 'https://dummyimage.com/1200x630/0c1a2a/ffffff.png&text=Quantalytics',
    navbar: {
      title: 'Quantalytics',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/example/quantalytics',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/getting-started',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/example/quantalytics',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Quantalytics Contributors.`,
    },
    prism: {
      theme: require('prism-react-renderer/themes/github'),
      darkTheme: require('prism-react-renderer/themes/dracula'),
    },
  },
};

module.exports = config;
