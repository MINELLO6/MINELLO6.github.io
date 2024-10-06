const ap = new APlayer({
    container: document.getElementById('aplayer'),
    fixed: true,
    autoplay: true, //自动播放
    lrcType: 3,
    audio: [{
            name: '踊り子',
            artist: 'Vaundy',
            url: 'https://github.com/MINELLO6/mymusic/blob/main/vaundy.mp3?raw=true',
            cover: 'https://y.qq.com/music/photo_new/T002R300x300M000002vI3qx4I4nHZ_2.jpg?max_age=2592000',
            lrc: 'https://github.com/MINELLO6/mymusic/blob/main/踊り子-Vaundy.lrc'
        },
        {
            name: 'Not Angry',
            artist: 'Chris James',
            url: 'https://github.com/MINELLO6/mymusic/blob/main/Not Angry.mp3?raw=true',
            cover: 'https://p2.music.126.net/8u9G6-hnZZV3HFZxU7RqHw==/109951165528742706.jpg?param=34y34',
            lrc: 'https://github.com/MINELLO6/mymusic/blob/main/2.lrc?raw=true'
        },
        {
            name: 'hill that i\'ll die on',
            artist: 'Jonah Kagen',
            url: 'https://github.com/MINELLO6/mymusic/blob/main/hill that i will die one.mp3?raw=true',
            cover: 'https://y.qq.com/music/photo_new/T002R300x300M000002WfJcX16BEzQ_1.jpg?max_age=2592000',
            lrc: 'https://github.com/MINELLO6/mymusic/blob/main/hill+that+ill+die+on-Jonah+Kagen.lrc?raw=true'
        },
        {
            name: '普通朋友',
            artist: '陶喆',
            url: 'https://github.com/MINELLO6/mymusic/blob/main/普通朋友.mp3?raw=true',
            cover: 'https://y.qq.com/music/photo_new/T002R300x300M000002PjDkE14OUkV_1.jpg?max_age=2592000',
            lrc: '[00:00.00] 我们一起学猫叫\n[99:99.99] 一起喵喵喵喵喵'
        }
    ]
});