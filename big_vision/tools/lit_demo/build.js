const sassPlugin = require('esbuild-sass-plugin').sassPlugin;

require('esbuild').serve({
  servedir: 'src',
  port: 8000,
}, {
  entryPoints: ['src/app.ts'],
  bundle: true,
  outfile: 'src/index.js',
  plugins: [
    sassPlugin({
      filter: /style.scss$/,
      type: 'style'
    }),
    sassPlugin({
      type: 'lit-css',
    }),
  ],
  sourcemap: true,
}).then(() => {
  console.log('Serving on port 8000');
}).catch(() => process.exit(1));
