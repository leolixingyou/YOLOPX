# 手机远程访问服务器指南

## 快速开始

### 方案1：Code Server（最推荐）
```bash
# 安装
curl -fsSL https://code-server.dev/install.sh | sh

# 直接启动（测试用）
code-server --host 0.0.0.0 --port 8080 --auth password

# 手机访问：http://服务器IP:8080
```

### 方案2：现成的在线IDE
- **GitHub Codespaces**：如果代码在GitHub上
- **Gitpod**：支持GitLab/GitHub/Bitbucket
- **Google Colab**：适合机器学习项目

## 安全建议

### 1. 使用nginx反向代理+SSL
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection upgrade;
    }
}
```

### 2. 使用SSH隧道（最安全）
在手机SSH客户端中：
```bash
# 端口转发
ssh -L 8080:localhost:8080 user@server-ip
# 然后手机浏览器访问 localhost:8080
```

### 3. 使用VPN
- WireGuard (轻量级)
- OpenVPN
- Tailscale (零配置VPN)

## 手机使用技巧

### 1. 浏览器选择
- **Chrome/Edge**：支持桌面模式
- **Firefox**：插件支持好

### 2. 键盘增强
- Android: Hacker's Keyboard
- iOS: 使用外接键盘

### 3. 手势操作
- Code Server支持触摸手势
- 双指缩放查看代码
- 长按呼出右键菜单

## 一键部署脚本

```bash
#!/bin/bash
# 保存为 mobile_setup.sh

# 检查是否安装code-server
if ! command -v code-server &> /dev/null; then
    echo "安装code-server..."
    curl -fsSL https://code-server.dev/install.sh | sh
fi

# 生成随机密码
PASSWORD=$(openssl rand -base64 12)

# 创建配置文件
mkdir -p ~/.config/code-server
cat > ~/.config/code-server/config.yaml <<EOF
bind-addr: 0.0.0.0:8080
auth: password
password: $PASSWORD
cert: false
EOF

# 启动服务
code-server &

echo "======================================="
echo "Code Server 已启动!"
echo "访问地址: http://$(hostname -I | awk '{print $1}'):8080"
echo "密码: $PASSWORD"
echo "======================================="
```

## 推荐工作流程

1. **轻量编辑**：使用SSH + nano/vim
2. **完整开发**：Code Server
3. **数据分析**：Jupyter Lab
4. **快速查看**：ttyd Web终端

## 故障排查

### 连接不上？
1. 检查防火墙：`sudo ufw allow 8080`
2. 检查服务状态：`ps aux | grep code-server`
3. 查看日志：`journalctl -u code-server`

### 性能优化
1. 降低分辨率
2. 关闭不必要的插件
3. 使用5G网络或WiFi

## 备用方案

如果以上都不行，可以试试：
1. **TeamViewer**：远程桌面
2. **AnyDesk**：轻量级远程桌面
3. **Chrome Remote Desktop**：谷歌远程桌面