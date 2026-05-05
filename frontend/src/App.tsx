import React, { useState, useEffect } from 'react';
import { 
  Layout, Card, Row, Col, Statistic, Form, InputNumber, Button, Typography, Space, Divider, Select, Badge, Tooltip as AntTooltip
} from 'antd';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area 
} from 'recharts';
import { 
  DashboardOutlined, ExperimentOutlined, RocketOutlined, ArrowUpOutlined, ArrowDownOutlined, DatabaseOutlined, RobotOutlined, CalendarOutlined, HistoryOutlined
} from '@ant-design/icons';
import { getPrediction, getSamples } from './services/api';
import { PriceFeatures, PredictionResponse } from './types';
import './App.css';

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;
const { Option } = Select;

const App: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [samples, setSamples] = useState<any[]>([]);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  const initialValues: PriceFeatures = {
    price_lag_0: 23000,
    brent_lag_0: 85.5,
    brent_lag_7: 82.0,
    brent_mean_30: 83.2,
    regime: 0,
    brent_diff_1: 0.5,
    brent_diff_7: 3.5,
    days_since_last_change: 5,
    model_name: 'random_forest',
    horizon: 1
  };

  useEffect(() => {
    // Load 100 recent samples on startup
    const loadSamples = async () => {
      try {
        const data = await getSamples();
        setSamples(data);
      } catch (err) {
        console.error("Failed to load samples", err);
      }
    };
    loadSamples();
  }, []);

  const handleSampleSelect = (value: string) => {
    const selected = samples.find(s => s.date === value);
    if (selected) {
      // Auto-fill form with sample data
      const newValues = {
        ...initialValues,
        ...selected,
        model_name: form.getFieldValue('model_name'),
        horizon: form.getFieldValue('horizon')
      };
      form.setFieldsValue(newValues);
    }
  };

  const onFinish = async (values: PriceFeatures) => {
    setLoading(true);
    try {
      const result = await getPrediction(values);
      setPrediction(result);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const getChartData = () => {
    const data = [
      { name: 'Last Period', price: 22800, type: 'Historical' },
      { name: 'Today', price: form.getFieldValue('price_lag_0') || initialValues.price_lag_0, type: 'Historical' },
    ];
    if (prediction && prediction.forecast) {
      prediction.forecast.forEach(pt => {
        data.push({ name: `Day +${pt.day}`, price: pt.predicted_price, type: 'Forecast' });
      });
    }
    return data;
  };

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      <Header style={{ background: '#001529', padding: '0 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Title level={3} style={{ color: '#fff', margin: 0 }}><RocketOutlined /> FuelPrice AI Enterprise</Title>
        <Badge status="processing" text={<span style={{color: '#fff'}}>Simulation Mode Active</span>} />
      </Header>
      
      <Layout>
        <Sider width={420} style={{ background: '#fff', padding: '24px', overflowY: 'auto' }}>
          <Title level={4}><HistoryOutlined /> Historical Context</Title>
          <Text type="secondary" style={{fontSize: '12px'}}>Select a day from the last 100 days to auto-fill the engine.</Text>
          <Select 
            placeholder="Pick a historical date..." 
            style={{ width: '100%', marginTop: '12px', marginBottom: '24px' }}
            onChange={handleSampleSelect}
            showSearch
          >
            {samples.map(s => (
              <Option key={s.date} value={s.date}>{s.date} (Price: {s.price_lag_0.toLocaleString()})</Option>
            ))}
          </Select>

          <Divider />
          <Title level={4}><ExperimentOutlined /> Prediction Engine</Title>
          <Form form={form} layout="vertical" initialValues={initialValues} onFinish={onFinish}>
            <Form.Item name="model_name" label={<strong>AI ENGINE</strong>}>
              <Select size="large">
                <Option value="random_forest">Random Forest (Stable)</Option>
                <Option value="xgboost">XGBoost (Trend Focus)</Option>
                <Option value="ensemble">Ensemble (Meta Model)</Option>
              </Select>
            </Form.Item>
            <Form.Item name="horizon" label={<strong>FORECAST HORIZON</strong>}>
              <Select size="large">
                <Option value={1}>1 Day</Option>
                <Option value={3}>3 Days</Option>
                <Option value={7}>7 Days (Weekly)</Option>
                <Option value={15}>15 Days</Option>
              </Select>
            </Form.Item>

            <Divider dashed>Features</Divider>
            <Row gutter={16}>
              {Object.keys(initialValues).filter(k => k !== 'model_name' && k !== 'horizon').map(key => (
                <Col span={12} key={key}>
                  <Form.Item name={key} label={<span style={{fontSize: '10px', color: '#8c8c8c'}}>{key.replace(/_/g, ' ').toUpperCase()}</span>}>
                    <InputNumber style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              ))}
            </Row>
            <Form.Item style={{ marginTop: '20px' }}>
              <Button type="primary" htmlType="submit" loading={loading} block size="large" icon={<DashboardOutlined />} style={{ height: '54px', borderRadius: '12px', background: '#001529' }}>
                RUN SIMULATION
              </Button>
            </Form.Item>
          </Form>
        </Sider>

        <Content style={{ padding: '24px' }}>
          <Row gutter={[24, 24]}>
            <Col span={8}>
              <Card bordered={false} hoverable><Statistic title="Simulation Base" value={form.getFieldValue('price_lag_0') || initialValues.price_lag_0} precision={2} valueStyle={{ color: '#52c41a' }} prefix={<DatabaseOutlined />} suffix="VND" /></Card>
            </Col>
            <Col span={8}>
              <Card bordered={false} style={{ background: '#e6f7ff' }} hoverable><Statistic title={<span>Target Projection</span>} value={prediction ? prediction.forecast[prediction.forecast.length - 1].predicted_price : 0} precision={2} valueStyle={{ color: '#1890ff', fontWeight: 'bold' }} suffix="VND" /></Card>
            </Col>
            <Col span={8}>
              <Card bordered={false} hoverable><Statistic title="Total Delta" value={prediction ? prediction.forecast.reduce((acc, curr) => acc + curr.predicted_delta, 0) : 0} precision={2} valueStyle={{ color: prediction && prediction.forecast.reduce((acc, curr) => acc + curr.predicted_delta, 0) > 0 ? '#ff4d4f' : '#52c41a' }} prefix={prediction && prediction.forecast.reduce((acc, curr) => acc + curr.predicted_delta, 0) > 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />} suffix="VND" /></Card>
            </Col>

            <Col span={24}>
              <Card title={<span><DashboardOutlined /> Simulation Visualization</span>} bordered={false} style={{ borderRadius: '16px' }}>
                <div style={{ height: '450px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={getChartData()}>
                        <defs><linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#1890ff" stopOpacity={0.2}/><stop offset="95%" stopColor="#1890ff" stopOpacity={0}/></linearGradient></defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                        <XAxis dataKey="name" axisLine={false} tickLine={false} />
                        <YAxis domain={['auto', 'auto']} axisLine={false} tickLine={false} tickFormatter={(val) => `${val.toLocaleString()}`} />
                        <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 24px rgba(0,0,0,0.12)' }} />
                        <Area type="monotone" dataKey="price" stroke="#1890ff" fillOpacity={1} fill="url(#colorPrice)" strokeWidth={4} />
                    </AreaChart>
                    </ResponsiveContainer>
                </div>
              </Card>
            </Col>
          </Row>
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;
