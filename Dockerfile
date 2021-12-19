FROM node:16

WORKDIR /usr/src/app

COPY package*.json ./

RUN apt-get update
RUN apt-get install git
RUN apt-get install wget python3 make gcc libc6-dev

RUN npm install

# RUN npm ci --only=production

COPY . .

EXPOSE 80

CMD [ "npm", "start" ]
