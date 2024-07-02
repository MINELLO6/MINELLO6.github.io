const ap = new APlayer({
    container: document.getElementById('aplayer'),
    fixed: true,
	autoplay: true, //自动播放
    audio: [
	{
        name: '踊り子',
        artist: 'Vaundy',
        url: 'https://lw-sycdn.kuwo.cn/1d593328e2e12b760e2d73371c8e2cca/6684f336/resource/30106/trackmedia/M50000029zVV22uiSW.mp3?bitrate$128&from=vip',
        cover: 'https://y.qq.com/music/photo_new/T002R300x300M000002vI3qx4I4nHZ_2.jpg?max_age=2592000',
    }, 
    {
        name: 'Not Angry',
        artist: 'Chris James',
        url: 'https://lw-sycdn.kuwo.cn/2cb309de38f08fa1b6ae214670c964d1/6684f363/resource/30106/trackmedia/M500002s3IOu1K3ztg.mp3?bitrate$128&from=vip',
        cover: 'https://p2.music.126.net/8u9G6-hnZZV3HFZxU7RqHw==/109951165528742706.jpg?param=34y34',
    }, 
    {
        name: 'hill that i\'ll die on',
        artist: 'Jonah Kagen',
        url: 'https://lw-sycdn.kuwo.cn/6d24e87604bbe86b7ff017033bfded95/6684f8ad/resource/30106/trackmedia/M500000NInng1DPeuY.mp3?bitrate$128&from=vip',
        cover: 'https://y.qq.com/music/photo_new/T002R300x300M000002WfJcX16BEzQ_1.jpg?max_age=2592000',
    }, 
    {
        name: '普通朋友',
        artist: '陶喆',
        url: 'https://lw-sycdn.kuwo.cn/39877e2514deb412cc947d0a250251b1/6684e8cd/resource/30106/trackmedia/M500001trdqY1wcxzH.mp3?bitrate$128&from=vip',
        cover: 'https://y.qq.com/music/photo_new/T002R300x300M000002PjDkE14OUkV_1.jpg?max_age=2592000',
    }
	]
});

