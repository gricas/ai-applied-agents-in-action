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
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

export interface ChatMessage {
  id: string;
  sender: 'agent' | 'user';
  message: string;
  timestamp?: string;
  category?: string;
}

const openingMessage: ChatMessage[] = [
  {
    id: '1',
    sender: 'agent',
    message: 'Hello! How can I assist you today?',
    timestamp: new Date().toISOString(),
  },
];

const ChatWindow = () => {
  const [inputValue, setInputValue] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>(openingMessage);
  const [loading, setLoading] = useState<boolean>(false);
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
    setMessages(openingMessage);
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

    const updatedMessages = [...messages, newMessage];
    setMessages(updatedMessages);
    setInputValue('');
    async function queryRaq() {
      setLoading(true);
      try {
        const result = await axios.post('/api/query/', {
          query: newMessage.message,
        });
        console.log(result.data.answer);

        const newAnswer: ChatMessage = {
          id: (updatedMessages.length + 1).toString(),
          sender: 'agent',
          message: result.data.answer,
          timestamp: new Date().toISOString(),
          category: result.data?.category,
        };

        setMessages([...updatedMessages, newAnswer]);
      } catch (err) {
        console.log(err);
      } finally {
        setLoading(false);
      }
    }
    queryRaq();
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
                {/*<p>{msg.message}</p>*/}
                <ReactMarkdown>{msg.message}</ReactMarkdown>
                <span
                  className={`chat-window__timestamp ${
                    msg.sender === 'user'
                      ? 'chat-window__timestamp--user'
                      : 'chat-window__timestamp--agent'
                  }`}
                >
                  {formatTimestamp(msg.timestamp)}
                  {msg.sender === 'agent' && msg.category && (
                    <span className='chat-window__category'>
                      {' '}
                      [ Category: {msg.category} ]
                    </span>
                  )}
                </span>
              </div>
            ))}
            {loading && (
              <div className='chat-window__loading'>
                <span className='chat-window__loading-ellipsis'>
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </span>
              </div>
            )}
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
            disabled={loading}
          />
        </div>
        <ButtonSet className='chat-window__button-set'>
          <Button
            kind='primary'
            renderIcon={Send}
            iconDescription='Send'
            onClick={handleSubmit}
            disabled={inputValue.length === 0 || loading}
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
            disabled={loading}
          >
            Reset
          </Button>
        </ButtonSet>
      </Column>
    </Grid>
  );
};

export default ChatWindow;
