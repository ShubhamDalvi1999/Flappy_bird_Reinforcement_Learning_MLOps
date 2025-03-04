import '../styles/globals.css';
import Head from 'next/head';

function MyApp({ Component, pageProps }) {
  return (
    <>
      <Head>
        <title>Flappy Bird Reinforcement Learning</title>
        <meta name="description" content="A Flappy Bird game trained using reinforcement learning" />
        <link rel="icon" href="/favicon.ico" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </Head>
      <Component {...pageProps} />
    </>
  );
}

export default MyApp; 