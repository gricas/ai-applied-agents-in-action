import { useState, ChangeEvent, KeyboardEvent, useRef, useEffect } from 'react';
import {
  Grid,
  Column,
  Tile,
  TextInput,
  Button,
  ButtonSet,
} from '@carbon/react';
import { Send, Reset } from '@carbon/icons-react';

export interface ChatMessage {
  id: string;
  sender: 'agent' | 'user';
  message: string;
  timestamp?: string;
}

const chatMessages: ChatMessage[] = [
  {
    id: '1',
    sender: 'agent',
    message: 'Hello! How can I assist you today?',
    timestamp: '2025-02-02T10:00:00Z',
  },
  {
    id: '2',
    sender: 'user',
    message: 'Can you help me understand my recent transactions?',
    timestamp: '2025-02-02T10:01:00Z',
  },
  {
    id: '3',
    sender: 'agent',
    message: 'Sure! Could you please provide more details?',
    timestamp: '2025-02-02T10:01:30Z',
  },
  {
    id: '4',
    sender: 'user',
    message:
      'I need information about the last three transactions on my account.',
    timestamp: '2025-02-02T10:02:00Z',
  },
  {
    id: '5',
    sender: 'agent',
    message: 'Here are the last three transactions:',
    timestamp: '2025-02-02T10:02:30Z',
  },
  {
    id: '6',
    sender: 'agent',
    message: '1. $200 at Grocery Store',
    timestamp: '2025-02-02T10:02:35Z',
  },
  {
    id: '7',
    sender: 'agent',
    message: '2. $50 at Gas Station',
    timestamp: '2025-02-02T10:02:40Z',
  },
  {
    id: '8',
    sender: 'agent',
    message: '3. $15 on Coffee',
    timestamp: '2025-02-02T10:02:45Z',
  },
  {
    id: '9',
    sender: 'user',
    message: 'Thanks! Can you tell me my current balance?',
    timestamp: '2025-02-02T10:03:00Z',
  },
  {
    id: '10',
    sender: 'agent',
    message: 'Your current balance is $3,245.67.',
    timestamp: '2025-02-02T10:03:15Z',
  },
];

const ChatWindow = () => {
  const [inputValue, setInputValue] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>(chatMessages);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>): void => {
    setInputValue(e.target.value);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  const handleReset = () => {
    setMessages([]);
    setInputValue('');
  };

  const handleSubmit = () => {
    if (!inputValue.trim()) return;

    const newMessage: ChatMessage = {
      id: (messages.length + 1).toString(),
      sender: 'user',
      message: inputValue,
      timestamp: new Date().toISOString(),
    };

    setMessages([...messages, newMessage]);
    setInputValue('');
  };

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return '';
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(timestamp));
  };

  return (
    <Grid fullWidth className='chat-window__grid'>
      <Column sm={4} md={8} lg={16} className='chat-window__column'>
        <Tile className='chat-window__tile'>
          <div className='chat-window__messages'>
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`chat-window__message ${
                  msg.sender === 'user'
                    ? 'chat-window__message--user'
                    : 'chat-window__message--agent'
                }`}
              >
                <p>{msg.message}</p>
                <span
                  className={`chat-window__timestamp ${
                    msg.sender === 'user'
                      ? 'chat-window__timestamp--user'
                      : 'chat-window__timestamp--agent'
                  }`}
                >
                  {formatTimestamp(msg.timestamp)}
                </span>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </Tile>
        <div className='chat-window__flex'>
          <TextInput
            className='chat-window__text-input'
            id='text-input-1'
            labelText='Chat with your docs!'
            placeholder='Type a message...'
            size='lg'
            type='text'
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
          />
        </div>
        <ButtonSet className='chat-window__button-set'>
          <Button
            kind='primary'
            renderIcon={Send}
            iconDescription='Send'
            onClick={handleSubmit}
            disabled={inputValue.length === 0}
            className='chat-window__button'
          >
            Send
          </Button>
          <Button
            kind='secondary'
            renderIcon={Reset}
            iconDescription='Reset'
            className='chat-window__button'
            onClick={handleReset}
          >
            Reset
          </Button>
        </ButtonSet>
      </Column>
    </Grid>
  );
};

export default ChatWindow;
