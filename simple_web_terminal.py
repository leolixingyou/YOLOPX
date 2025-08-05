#!/usr/bin/env python3
"""
简单的Web终端服务器
可以通过浏览器访问命令行
"""
import os
import pty
import select
import subprocess
import struct
import fcntl
import termios
import signal
import sys
import json
from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Web Terminal - YOLOPX</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/xterm/4.19.0/xterm.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/xterm/4.19.0/xterm.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/xterm/4.19.0/addons/fit/fit.js"></script>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 10px;
            background: #1e1e1e;
            font-family: monospace;
        }
        #terminal {
            width: 100%;
            height: calc(100vh - 60px);
        }
        .header {
            color: #fff;
            padding: 10px;
            background: #007ACC;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .status {
            color: #0f0;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h3 style="margin: 0;">YOLOPX Web Terminal</h3>
        <div class="status">Connected to: {{ hostname }}</div>
    </div>
    <div id="terminal"></div>
    
    <script>
        const term = new Terminal({
            cursorBlink: true,
            fontSize: 14,
            fontFamily: 'Consolas, Monaco, monospace',
            theme: {
                background: '#1e1e1e',
                foreground: '#d4d4d4'
            }
        });
        
        const socket = io();
        
        term.open(document.getElementById('terminal'));
        term.fit();
        
        term.onData(data => {
            socket.emit('input', data);
        });
        
        socket.on('output', data => {
            term.write(data);
        });
        
        socket.on('connect', () => {
            term.write('\\r\\n*** 已连接到服务器 ***\\r\\n');
        });
        
        socket.on('disconnect', () => {
            term.write('\\r\\n*** 连接已断开 ***\\r\\n');
        });
        
        // 自适应大小
        window.addEventListener('resize', () => {
            term.fit();
        });
    </script>
</body>
</html>
'''

# 存储会话
sessions = {}

@app.route('/')
def index():
    hostname = subprocess.check_output(['hostname']).decode().strip()
    return render_template_string(HTML_TEMPLATE, hostname=hostname)

@socketio.on('connect')
def handle_connect():
    """处理新连接"""
    # 创建伪终端
    (child_pid, fd) = pty.fork()
    if child_pid == 0:
        # 子进程：运行shell
        subprocess.run(os.environ.get('SHELL', '/bin/bash'))
    else:
        # 父进程：保存会话信息
        sessions[request.sid] = {
            'fd': fd,
            'pid': child_pid
        }
        
        # 设置非阻塞
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        
        print(f"新连接: {request.sid}")
        
        # 开始读取输出
        socketio.start_background_task(target=read_output, sid=request.sid, fd=fd)

@socketio.on('disconnect')
def handle_disconnect():
    """处理断开连接"""
    if request.sid in sessions:
        session = sessions[request.sid]
        os.close(session['fd'])
        os.kill(session['pid'], signal.SIGTERM)
        del sessions[request.sid]
        print(f"断开连接: {request.sid}")

@socketio.on('input')
def handle_input(data):
    """处理输入"""
    if request.sid in sessions:
        fd = sessions[request.sid]['fd']
        os.write(fd, data.encode())

def read_output(sid, fd):
    """读取终端输出"""
    while sid in sessions:
        socketio.sleep(0.01)
        try:
            output = os.read(fd, 1024).decode()
            if output:
                socketio.emit('output', output, room=sid)
        except:
            pass

if __name__ == '__main__':
    print("启动Web终端服务器...")
    print("访问地址: http://0.0.0.0:5000")
    print("按 Ctrl+C 停止服务器")
    
    # 安装依赖提示
    try:
        import flask_socketio
    except ImportError:
        print("\n请先安装依赖:")
        print("pip install flask flask-socketio python-socketio")
        sys.exit(1)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)